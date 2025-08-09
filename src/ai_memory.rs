use rusqlite::{params, Connection};
use chrono::{Utc, DateTime};
use serde::{Serialize, Deserialize};
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatTurn {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub ts: DateTime<Utc>,
}

pub struct Memory { conn: Connection }

impl Memory {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS memory(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 role TEXT NOT NULL,
                 content TEXT NOT NULL,
                 ts TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory(ts);
             COMMIT;",
        )?;
        Ok(Self { conn })
    }
    pub fn append(&self, role: &str, content: &str) -> Result<()> {
        let ts = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO memory (role, content, ts) VALUES (?1, ?2, ?3)",
            params![role, content, ts],
        )?;
        Ok(())
    }
    pub fn last_turns(&self, limit: usize) -> Result<Vec<ChatTurn>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, role, content, ts FROM memory ORDER BY id DESC LIMIT ?"
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            let ts_str: String = r.get(3)?;
            let ts = DateTime::parse_from_rfc3339(&ts_str).unwrap().with_timezone(&Utc);
            Ok(ChatTurn { id: r.get(0)?, role: r.get(1)?, content: r.get(2)?, ts })
        })?;
        let mut v: Vec<ChatTurn> = rows.filter_map(|r| r.ok()).collect();
        v.reverse();
        Ok(v)
    }
    pub fn clear(&self) -> Result<()> { self.conn.execute("DELETE FROM memory", params![])?; Ok(()) }
}