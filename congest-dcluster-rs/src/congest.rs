//! A synchronous CONGEST-KT1 network.
//!
//! Section 2.2 of the paper: computation proceeds in synchronous rounds, in
//! each round a vertex may send one message of `O(log n)` bits along each
//! incident edge, identifiers fit in `O(log n)` bits, and under KT1 every
//! vertex knows the identifiers of its neighbors before the first round.
//!
//! The bandwidth limit is enforced rather than assumed. A payload is a tag plus
//! at most [`MAX_WORDS`] words, and every word has to be an identifier or a
//! counter in `[0, max(n, 2))`. That is `O(1)` fields of `O(log n)` bits each,
//! so a payload that passes [`Network::send`] is one the model allows. Anything
//! wider is rejected instead of silently accepted, which is the point: a
//! simulator that lets you ship a whole adjacency list in one message is not
//! simulating CONGEST.
//!
//! `send` also refuses to deliver along a non-edge, so a vertex can only learn
//! about the graph through its incident edges.

use crate::graph::Graph;

/// Most words a single message may carry.
pub const MAX_WORDS: usize = 3;

/// Message kinds used by the algorithms in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    /// Line 5 of Algorithm 1, broadcast by a sampled center: "I'm a center".
    CenterAnnounce,
    /// Line 9 reply: the sender is joining the receiver's cluster.
    Join,
    /// Line 11, a probe to a neighbor sampled from `N_G(v)`.
    Probe,
    /// Line 12 reply: whether the sender has left `G_i`, and which cluster it
    /// is in if any.
    ProbeReply,
    /// Lines 13 and 14: the sender attached itself to the receiver's cluster.
    Link,
    /// Step 2 of Algorithm Partition: the sender put this edge in `J`.
    EdgeSampled,
    /// Only used by tests.
    Test,
}

/// A tag plus a few small words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Payload {
    pub tag: Tag,
    pub words: Vec<u64>,
}

impl Payload {
    pub fn new(tag: Tag, words: &[u64]) -> Payload {
        Payload { tag, words: words.to_vec() }
    }

    pub fn word(&self, i: usize) -> u64 {
        self.words[i]
    }
}

/// A message in flight, from the receiver's point of view.
#[derive(Clone, Debug)]
pub struct Message {
    pub from: u32,
    pub payload: Payload,
}

/// Why a send was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SendError {
    /// More than [`MAX_WORDS`] words, so the payload is not `O(log n)` bits.
    TooManyWords { got: usize, limit: usize },
    /// A word too large to be an identifier or a counter on `n` vertices.
    WordOutOfRange { word: u64, limit: u64 },
    /// The sender and receiver are not adjacent in `G`.
    NotAnEdge { from: u32, to: u32 },
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::TooManyWords { got, limit } => {
                write!(f, "payload carries {got} words, the limit is {limit}")
            }
            SendError::WordOutOfRange { word, limit } => {
                write!(f, "word {word} does not fit in an O(log n)-bit field (limit {limit})")
            }
            SendError::NotAnEdge { from, to } => write!(f, "{from} and {to} are not adjacent"),
        }
    }
}

impl std::error::Error for SendError {}

/// The network. Owns the message and round counters.
pub struct Network<'g> {
    graph: &'g Graph,
    pending: Vec<Vec<Message>>,
    inbox: Vec<Vec<Message>>,
    rounds: u64,
    messages: u64,
    word_limit: u64,
}

impl<'g> Network<'g> {
    pub fn new(graph: &'g Graph) -> Network<'g> {
        let n = graph.n();
        Network {
            graph,
            pending: vec![Vec::new(); n],
            inbox: vec![Vec::new(); n],
            rounds: 0,
            messages: 0,
            // An identifier lives in [0, n) and the flags and counters this
            // crate sends never exceed that, so ceil(log2 max(n, 2)) bits are
            // enough for any word.
            word_limit: (n as u64).max(2),
        }
    }

    pub fn graph(&self) -> &'g Graph {
        self.graph
    }

    /// Rounds elapsed so far.
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Messages accepted so far. This is the quantity Lemma 4.2 bounds.
    pub fn messages(&self) -> u64 {
        self.messages
    }

    /// Queue one message for delivery at the start of the next round.
    ///
    /// The payload is checked here, so an oversized message never gets counted
    /// and never gets delivered.
    pub fn send(&mut self, from: usize, to: usize, payload: Payload) -> Result<(), SendError> {
        if payload.words.len() > MAX_WORDS {
            return Err(SendError::TooManyWords { got: payload.words.len(), limit: MAX_WORDS });
        }
        for &w in &payload.words {
            if w >= self.word_limit {
                return Err(SendError::WordOutOfRange { word: w, limit: self.word_limit });
            }
        }
        if !self.graph.is_neighbor(from, to) {
            return Err(SendError::NotAnEdge { from: from as u32, to: to as u32 });
        }
        self.pending[to].push(Message { from: from as u32, payload });
        self.messages += 1;
        Ok(())
    }

    /// Deliver everything queued and step the clock.
    ///
    /// The round counter advances whether or not anything was sent. DCluster
    /// has a fixed round structure that does not depend on the input, so an
    /// idle round still costs a round.
    pub fn advance_round(&mut self) {
        for v in 0..self.inbox.len() {
            self.inbox[v].clear();
            std::mem::swap(&mut self.inbox[v], &mut self.pending[v]);
        }
        self.rounds += 1;
    }

    /// What `v` received at the start of the current round.
    pub fn inbox(&self, v: usize) -> &[Message] {
        &self.inbox[v]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph;

    #[test]
    fn oversized_payload_is_rejected() {
        let g = graph::clique(8);
        let mut net = Network::new(&g);
        let too_wide = Payload::new(Tag::Test, &[1, 2, 3, 4]);
        assert_eq!(
            net.send(0, 1, too_wide),
            Err(SendError::TooManyWords { got: 4, limit: MAX_WORDS })
        );
        // A rejected send is not delivered and is not charged.
        assert_eq!(net.messages(), 0);
        net.advance_round();
        assert!(net.inbox(1).is_empty());
    }

    #[test]
    fn word_wider_than_log_n_bits_is_rejected() {
        let g = graph::clique(8);
        let mut net = Network::new(&g);
        // 8 vertices, so a word has to be under 8. Packing two identifiers into
        // one word is the obvious way to cheat and it does not pass.
        assert_eq!(
            net.send(0, 1, Payload::new(Tag::Test, &[3 * 8 + 5])),
            Err(SendError::WordOutOfRange { word: 29, limit: 8 })
        );
        assert!(net.send(0, 1, Payload::new(Tag::Test, &[7])).is_ok());
        assert_eq!(net.messages(), 1);
    }

    #[test]
    fn sending_along_a_non_edge_is_rejected() {
        let g = graph::path(4);
        let mut net = Network::new(&g);
        assert_eq!(
            net.send(0, 3, Payload::new(Tag::Test, &[])),
            Err(SendError::NotAnEdge { from: 0, to: 3 })
        );
        assert!(net.send(0, 1, Payload::new(Tag::Test, &[])).is_ok());
    }

    #[test]
    fn delivery_happens_at_the_next_round() {
        let g = graph::path(3);
        let mut net = Network::new(&g);
        net.send(0, 1, Payload::new(Tag::Test, &[2])).unwrap();
        assert!(net.inbox(1).is_empty(), "not delivered inside the sending round");
        net.advance_round();
        assert_eq!(net.inbox(1).len(), 1);
        assert_eq!(net.inbox(1)[0].from, 0);
        assert_eq!(net.inbox(1)[0].payload.word(0), 2);
        net.advance_round();
        assert!(net.inbox(1).is_empty(), "inboxes do not persist across rounds");
        assert_eq!(net.rounds(), 2);
        assert_eq!(net.messages(), 1);
    }
}
