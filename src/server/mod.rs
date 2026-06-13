//! HTTP/gRPC server — port of rust/src/server/
//!
//! OpenAI-compatible API server.

use axum::{Router, routing::post, Json};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub fn build_router() -> Router {
    Router::new()
        .route("/v1/completions", post(handle_completion))
        .route("/health", post(handle_health))
}

async fn handle_health() -> &'static str {
    "ok"
}

async fn handle_completion(Json(_req): Json<CompletionRequest>) -> Json<CompletionResponse> {
    Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        choices: vec![Choice {
            text: String::new(),
            index: 0,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    })
}
