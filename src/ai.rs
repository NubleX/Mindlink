use crate::ai_memory::Memory;
use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use rand::{thread_rng, Rng};
use reqwest::{header, Client};
use reqwest_eventsource::EventSource;
use serde::{Deserialize, Serialize};
use std::env;
use tokio::time::{sleep, Duration};

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIMessage {
    role: String,   // "user" | "assistant" | "system"
    content: String,
}

#[derive(Serialize, Deserialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct StreamChunkChoiceDelta {
    content: Option<String>,
    // role is present in the schema but not needed; keep to avoid schema drift warnings
    #[allow(dead_code)]
    role: Option<String>,
}

#[derive(Deserialize)]
struct StreamChunkChoice {
    delta: StreamChunkChoiceDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChunkChoice>,
}

pub struct AiAgent {
    provider: String,     // "openai"
    model: String,        // "gpt-5"
    client: Client,
    mem: Memory,
    memory_turns: usize,
    #[allow(dead_code)]
    project_mode: bool,
}

impl AiAgent {
    pub fn new(memory_path: &str, project_mode: bool) -> Result<Self> {
        dotenvy::dotenv().ok();
        let provider = env::var("AI_PROVIDER").unwrap_or_else(|_| "openai".into());
        let model = env::var("AI_MODEL").unwrap_or_else(|_| "gpt-5".into());
        let client = Client::builder().build()?;
        let mem = Memory::open(memory_path)?;
        let memory_turns = env::var("AI_MEMORY_TURNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);

        Ok(Self {
            provider,
            model,
            client,
            mem,
            memory_turns,
            project_mode,
        })
    }

    fn build_history(&self) -> Result<Vec<OpenAIMessage>> {
        let history = self.mem.last_turns(self.memory_turns)?;
        let mut msgs = Vec::with_capacity(history.len());
        for h in history {
            msgs.push(OpenAIMessage {
                role: h.role,
                content: h.content,
            });
        }
        Ok(msgs)
    }

    pub async fn ask_streaming(&self, user_prompt: &str) -> Result<String> {
        if self.provider != "openai" {
            return Err(anyhow!("Only 'openai' provider is enabled in this build."));
        }
        let api_key =
            env::var("OPENAI_API_KEY").map_err(|_| anyhow!("OPENAI_API_KEY not set"))?;

        let mut messages = self.build_history()?;
        messages.push(OpenAIMessage {
            role: "user".into(),
            content: user_prompt.into(),
        });

        let req = OpenAIRequest {
            model: self.model.clone(),
            messages,
            stream: true,
        };

        let max_retries: usize = env::var("AI_MAX_RETRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let base_backoff_ms: u64 = env::var("AI_BACKOFF_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        let mut attempts = 0usize;
        let mut acc = String::new();

        loop {
            attempts += 1;

            let req_builder = self
                .client
                .post("https://api.openai.com/v1/chat/completions")
                .header(header::AUTHORIZATION, format!("Bearer {}", api_key))
                .header(header::CONTENT_TYPE, "application/json")
                .json(&req);

            let mut es = match EventSource::new(req_builder) {
                Ok(es) => es,
                Err(e) => {
                    let msg = e.to_string();
                    if (msg.contains("429") || msg.contains("Too Many Requests"))
                        && attempts <= max_retries
                    {
                        let jitter: u64 = thread_rng().gen_range(0..250);
                        let backoff =
                            Duration::from_millis(base_backoff_ms * attempts as u64 + jitter);
                        eprintln!("rate limited (429), retrying in {:?}...", backoff);
                        sleep(backoff).await;
                        continue;
                    }
                    if attempts > max_retries {
                        eprintln!(
                            "stream failed after {} attempts; falling back to non-stream.",
                            attempts - 1
                        );
                        let out = self.ask_once(user_prompt).await?;
                        println!("{}", out);
                        return Ok(out);
                    }
                    return Err(anyhow!(e));
                }
            };

            while let Some(event) = es.next().await {
                match event {
                    Ok(reqwest_eventsource::Event::Open) => {
                        // connected; nothing to print
                    }
                    Ok(reqwest_eventsource::Event::Message(msg)) => {
                        let data = msg.data.trim();
                        if data == "[DONE]" {
                            es.close();
                            break;
                        }
                        if let Ok(payload) = serde_json::from_str::<StreamChunk>(data) {
                            if let Some(choice) = payload.choices.get(0) {
                                if let Some(piece) = &choice.delta.content {
                                    print!("{}", piece);
                                    acc.push_str(piece);
                                    use std::io::Write;
                                    let _ = std::io::stdout().flush();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        es.close();
                        if (msg.contains("429") || msg.contains("Too Many Requests"))
                            && attempts <= max_retries
                        {
                            let jitter: u64 = thread_rng().gen_range(0..250);
                            let backoff = Duration::from_millis(
                                base_backoff_ms * attempts as u64 + jitter,
                            );
                            eprintln!("\nstream 429, retrying in {:?}...", backoff);
                            sleep(backoff).await;
                            acc.clear();
                            continue;
                        }
                        if attempts > max_retries {
                            eprintln!(
                                "stream failed after {} attempts; falling back to non-stream.",
                                attempts - 1
                            );
                            let out = self.ask_once(user_prompt).await?;
                            println!("{}", out);
                            self.mem.append("user", user_prompt)?;
                            self.mem.append("assistant", &out)?;
                            println!();
                            return Ok(out);
                        }
                        return Err(anyhow!("stream error: {}", msg));
                    }
                }
            }

            // success
            break;
        }

        self.mem.append("user", user_prompt)?;
        self.mem.append("assistant", &acc)?;
        println!();
        Ok(acc)
    }

    // Non-stream fallback
    pub async fn ask_once(&self, user_prompt: &str) -> Result<String> {
        let api_key = env::var("OPENAI_API_KEY")?;
        let mut messages = self.build_history()?;
        messages.push(OpenAIMessage {
            role: "user".into(),
            content: user_prompt.into(),
        });

        #[derive(Serialize)]
        struct Req {
            model: String,
            messages: Vec<OpenAIMessage>,
            stream: bool,
        }
        #[derive(Deserialize)]
        struct RespChoice {
            message: OpenAIMessage,
        }
        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<RespChoice>,
        }

        let req = Req {
            model: self.model.clone(),
            messages,
            stream: false,
        };

        let res: Resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header(header::AUTHORIZATION, format!("Bearer {}", api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&req)
            .send()
            .await?
            .json()
            .await?;

        let out = res
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        self.mem.append("user", user_prompt)?;
        self.mem.append("assistant", &out)?;
        Ok(out)
    }

    pub fn memory_show(&self, limit: usize) -> Result<Vec<crate::ai_memory::ChatTurn>> {
        self.mem.last_turns(limit)
    }

    pub fn memory_clear(&self) -> Result<()> {
        self.mem.clear()
    }
}