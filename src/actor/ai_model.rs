#![allow(unused)]

use steady_state::*;
use crate::includes::llm_engine::LlmEngine;
use crate::includes::file_utils::FileMetadata;
use crate::includes::config::*;
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
// const MODEL_FILE_PATH: &str = "/home/zaigiaz/third_party/ai_models/Qwen/Qwen3-4B-Instruct-2507-Q8_0.gguf";


/// run function for the AI model actor
pub async fn run(actor: SteadyActorShadow, ai_rx: SteadyRx<FileMetadata>) -> Result<(),Box<dyn Error>> {

    let actor = actor.into_spotlight([&ai_rx], []);
	
	if actor.use_internal_behavior {
	    internal_behavior(actor, ai_rx).await
	} else {
	    println!("error in the internal actor function for ai actor"); 
	    process::exit(0x0100);
	} 
}

/// Internal behaviour for the actor
async fn internal_behavior<A: SteadyActor>(mut actor: A, ai_rx: SteadyRx<FileMetadata>) -> Result<(),Box<dyn Error>> {

    let ConfigStruct: Config = read_toml("./config.toml")?;
    let MODEL_FILE_PATH: &str = ConfigStruct.ai_model.model_path.as_str();

    // load the AI model and run;
    let engine = LlmEngine::load_new_model(
        MODEL_FILE_PATH
    )?;
    
    let mut ai_rx = ai_rx.lock().await;
    
    while actor.is_running(|| ai_rx.is_closed_and_empty()) {

    /// TODO :: write decision to file, but later, to TUI
    /// TODO :: LLM should provide one word response, we could check with regex to make sure, then re-prompt if outlier?

    await_for_all!(actor.wait_avail(&mut ai_rx, 1));

    // take the message from the channel and then turn the metadata into a prompt message for the channel

    /// TODO :: Cant figure out why I am getting actor panics from this right now.
    let recieved = actor.try_take(&mut ai_rx).expect("recieving metadata from DB, AI Model <- Database");
    let metadata_prompt_message = recieved.to_prompt()?.to_string();
    println!("{}", metadata_prompt_message);

    let resp = engine.infer_model(&metadata_prompt_message)?;
    println!("Here is the AI reponse: {}", resp);
    }

    actor.request_shutdown().await;

    return Ok(());
} 


