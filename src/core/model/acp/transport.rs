//! In-memory newline transport for unit tests (simulates ACP stdio).

use std::io::{self, Read, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// One end of a pair of connected fake stdio streams: lines written here are read by the peer.
#[derive(Debug)]
pub struct MemoryLineEnd {
    inbox: Receiver<String>,
    outbox: Sender<String>,
}

impl MemoryLineEnd {
    pub fn send_line(&self, line: impl Into<String>) -> Result<(), mpsc::SendError<String>> {
        let mut s = line.into();
        if !s.ends_with('\n') {
            s.push('\n');
        }
        self.outbox.send(s)
    }

    pub fn recv_line_timeout(&self, timeout: Duration) -> Result<String, RecvLineError> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(RecvLineError::Timeout);
            }
            match self
                .inbox
                .recv_timeout(remaining.min(Duration::from_millis(50)))
            {
                Ok(s) => return Ok(s.trim_end_matches(['\r', '\n']).to_string()),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(RecvLineError::Disconnected);
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecvLineError {
    Timeout,
    Disconnected,
}

/// Create two ends; writing on `a` is received on `b` and vice versa.
pub fn memory_line_pair() -> (MemoryLineEnd, MemoryLineEnd) {
    let (a_to_b_tx, a_to_b_rx) = mpsc::channel::<String>();
    let (b_to_a_tx, b_to_a_rx) = mpsc::channel::<String>();
    let a = MemoryLineEnd {
        inbox: b_to_a_rx,
        outbox: a_to_b_tx,
    };
    let b = MemoryLineEnd {
        inbox: a_to_b_rx,
        outbox: b_to_a_tx,
    };
    (a, b)
}

/// Feeds [`BufReader`] one line at a time from a channel (blocking `recv`).
pub struct ChannelReader {
    rx: Receiver<String>,
    buf: Vec<u8>,
    pos: usize,
}

impl ChannelReader {
    pub fn new(rx: Receiver<String>) -> Self {
        Self {
            rx,
            buf: Vec::new(),
            pos: 0,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(mut line) => {
                    if !line.ends_with('\n') {
                        line.push('\n');
                    }
                    self.buf = line.into_bytes();
                    self.pos = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Buffers writes until a newline, then sends one logical line to the peer.
pub struct LineBufferWriter {
    tx: Sender<String>,
    acc: Vec<u8>,
}

impl LineBufferWriter {
    pub fn new(tx: Sender<String>) -> Self {
        Self {
            tx,
            acc: Vec::new(),
        }
    }

    fn drain_lines(&mut self) -> io::Result<()> {
        while let Some(i) = self.acc.iter().position(|&b| b == b'\n') {
            let rest = self.acc[i + 1..].to_vec();
            let line_bytes = self.acc[..i].to_vec();
            self.acc = rest;
            if line_bytes.is_empty() {
                continue;
            }
            let s = String::from_utf8(line_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            self.tx
                .send(s)
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "ACP peer disconnected"))?;
        }
        Ok(())
    }
}

impl Write for LineBufferWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.acc.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.drain_lines()
    }
}

pub struct LineTransport {
    pub reader: ChannelReader,
    pub writer: LineBufferWriter,
}

/// Two connected ends: bytes written on one side are read as lines on the other.
pub fn line_channel_pair() -> (LineTransport, LineTransport) {
    let (c_to_a_tx, c_to_a_rx) = mpsc::channel::<String>();
    let (a_to_c_tx, a_to_c_rx) = mpsc::channel::<String>();
    let client = LineTransport {
        reader: ChannelReader::new(a_to_c_rx),
        writer: LineBufferWriter::new(c_to_a_tx),
    };
    let agent = LineTransport {
        reader: ChannelReader::new(c_to_a_rx),
        writer: LineBufferWriter::new(a_to_c_tx),
    };
    (client, agent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_roundtrip_line() {
        let (left, right) = memory_line_pair();
        left.send_line(r#"{"x":1}"#).unwrap();
        let got = right
            .recv_line_timeout(Duration::from_secs(1))
            .expect("recv");
        assert_eq!(got, r#"{"x":1}"#);
    }

    #[test]
    fn line_channel_pair_roundtrip_via_read_write() {
        let (mut client, mut agent) = line_channel_pair();
        writeln!(client.writer, r#"{{"hi":true}}"#).unwrap();
        client.writer.flush().unwrap();
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(&mut agent.reader), &mut line)
            .unwrap();
        assert!(line.trim().contains("\"hi\":true"));
    }
}
