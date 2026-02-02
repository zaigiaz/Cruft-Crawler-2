#![allow(unused)]

use steady_state::*;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::io::Write;
use std::num::NonZeroU32;
use std::{any, fs};

use crate::actor::crawler::FileMeta;

// run function 
pub async fn run(actor: SteadyActorShadow, crawler_to_model_rx: SteadyRx<FileMeta>) -> Result<(),Box<dyn Error>> {

    let actor = actor.into_spotlight([&crawler_to_model_rx], []);

	if actor.use_internal_behavior {
	    internal_behavior(actor, crawler_to_model_rx).await
	} else {
	    actor.simulated_behavior(vec!(&crawler_to_model_rx)).await
	}
}

// code to run our AI model through the Llama2 Crate
fn run_model()-> anyhow::Result<()>{

    // Initialize the backend
    let backend = LlamaBackend::init()?;
    
    // Set up model parameters
    let model_params = LlamaModelParams::default();
    
    // TODO: change this to be relative to our actual project
    let model_file_path  = "../models/smollm3-3b-q4_k_m.gguf";
    let prompt_file_path = "../../data/model_data/prompt.txt";
    
    // Load the model
    let model = LlamaModel::load_from_file(
	&backend,
	&model_file_path,
	&model_params
    )?;
    
    // Create context with 2048 token context size
    let ctx_params = LlamaContextParams::default()
	.with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
    
    let mut ctx = model.new_context(&backend, ctx_params)?;
    
    // The prompt
    let prompt = fs::read_to_string(prompt_file_path)?;
    
    // Tokenize the prompt
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    
    println!("Prompt: {}", prompt);
    println!("Generating response...\n");
    
    // Create a batch and add all prompt tokens
    let mut batch = LlamaBatch::new(512, 1);
    let last_index = (tokens.len() - 1) as i32;
    
    for (i, token) in tokens.into_iter().enumerate() {
	let is_last = i as i32 == last_index;
	batch.add(token, i as i32, &[0], is_last)?;
    }
    
    // Process the prompt
    ctx.decode(&mut batch)?;
    
    // Set up sampler (greedy sampling - always picks most likely token)
    let mut sampler = LlamaSampler::chain_simple([
	LlamaSampler::dist(10), // seed
	//LlamaSampler::greedy(),
    ]);
    
    // Generate tokens
    let max_tokens = 100;
    let mut n_cur = batch.n_tokens();
    
    // Decoder for handling UTF-8 properly
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    
    for _ in 0..max_tokens {
	// Sample the next token
	let token = sampler.sample(&ctx, batch.n_tokens() - 1);
	sampler.accept(token);
	
	// Check for end of generation
	if model.is_eog_token(token) {
	    println!();
	    break;
	}
	
	// Convert token to bytes and then to string
	let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
	let mut output_string = String::with_capacity(32);
	decoder.decode_to_string(&output_bytes, &mut output_string, false);
	
	print!("{}", output_string);
	// Write responses to output file
	let mut file = fs::OpenOptions::new()
	    .append(true)   // append mode
	    .create(true)   // create if it doesn't exist
	    .open("../../data/output.txt")
	
	writeln!(file, "{}", output_string)?;

	std::io::stdout().flush()?;

	
	// Prepare next iteration
	batch.clear();
	batch.add(token, n_cur, &[0], true)?;
	n_cur += 1;
	
	ctx.decode(&mut batch)?;
    }
    
    println!("\n");
    
    Ok(())
}

// Internal behaviour for the actor
async fn internal_behavior<A: SteadyActor>(mut actor: A, crawler_to_ai_model_rx: SteadyRx<FileMeta>) -> Result<(),Box<dyn Error>> {
	
    let mut crawler_to_ai_model_rx = crawler_to_ai_model_rx.lock().await;
	    
    while actor.is_running(|| crawler_to_ai_model_rx.is_closed_and_empty() || ai_model_to_ui_tx.mark_closed()) {
		
	// run our AI model that is in the model/ folder
	run_model();

	// Recieving data from crawler actor
	actor.wait_avail(&mut crawler_to_ai_model_rx, 1).await;
        let recieved = actor.try_take(&mut crawler_to_ai_model_rx);
	let message = recieved.expect("Expected a string");			
    } 

	return Ok(());
}
