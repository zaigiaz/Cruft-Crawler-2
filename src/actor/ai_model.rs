#![allow(unused)]

use steady_state::*;
use crate::includes::llm_engine::LlmEngine;
use std::process;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;
use std::io::Write;
use std::num::NonZeroU32;
use std::{any, fs};

// TODO :: make this into an option for toml configuration file
const MODEL_FILE_PATH: &str  = "/home/zaigiaz/third_party/ai_models/Qwen/Qwen3-4B-Instruct-2507-Q8_0.gguf";

/// run function for the AI model actor
pub async fn run(actor: SteadyActorShadow) -> Result<(),Box<dyn Error>> {

    let actor = actor.into_spotlight([], []);
	
	if actor.use_internal_behavior {
	    internal_behavior(actor).await
	} else {
	    println!("error in the internal actor function for ai actor"); 
	    process::exit(0x0100);
	} 
}

/// Internal behaviour for the actor
async fn internal_behavior<A: SteadyActor>(mut actor: A) -> Result<(),Box<dyn Error>> {

    // load the AI model and run;
    /// TODO :: figure out why not running
	let engine = LlmEngine::load_new_model(
        MODEL_FILE_PATH
    )?;
    

    let initial_prompt = "Hello, you are an AI model, please provide basic response, like a greeting statement, short and concise.";

    /// TODO :: think about new condition for while loop
    // while actor.is_running() {
    // }

    /// TODO :: write decision to file, but later, to TUI
    /// TODO :: LLM should provide one word response, we could check with regex to make sure, then re-prompt if outlier?
    
    let resp = engine.infer_model(&initial_prompt)?;
    
    println!("{:?}", resp);

    actor.request_shutdown().await;

    return Ok(());
} 


