// leave this here for now, until cleanup time
#![allow(warnings)]

use steady_state::*;
use std::time::Duration;
use crate::includes::file_utils::FileMetadata;

// crate that adds in both the actors from the actor/ directory
pub(crate) mod actor {  
    pub(crate) mod crawler;
    pub(crate) mod db_manager;    
    pub(crate) mod ai_model;    
}

pub(crate) mod includes {
    pub(crate) mod file_utils;
    pub(crate) mod db_utils;
    pub(crate) mod write_ahead_log;
    pub(crate) mod env;
    pub(crate) mod llm_engine;
    pub(crate) mod config;
}

fn main() -> Result<(), Box<dyn Error>> {

    init_logging(LogLevel::Info, None)?;   

    // pass unit value into .build() to ignore cli_args for now
    let mut graph = GraphBuilder::default().build(());

    build_graph(&mut graph); 

    graph.start();  

    graph.block_until_stopped(Duration::from_secs(1)) 
}

const NAME_CRAWLER: &str = "CRAWLER";
const NAME_DB: &str = "DB_MANAGER";
const NAME_AI: &str = "AI_MODEL";


fn build_graph(graph: &mut Graph) {

    // build channels and configure colors on graph if they fill up too much
    let channel_builder = graph.channel_builder()
        .with_filled_trigger(Trigger::AvgAbove(Filled::p90()), AlertColor::Red) 
        .with_filled_trigger(Trigger::AvgAbove(Filled::p60()), AlertColor::Orange)
        .with_filled_percentile(Percentile::p80());

    // Build Channels for Sender and Reciever Tx and Rx for communication between actors
    let (crawler_tx, crawler_rx)  = channel_builder.build();
    let (ai_tx, ai_rx)            = channel_builder.build();
    
    // build actor interface
    let actor_builder = graph.actor_builder()
        .with_load_avg()
        .with_mcpu_avg();

    // crawler actor
    let state = new_state();
    actor_builder.with_name(NAME_CRAWLER)
        .build(move |actor| actor::crawler::run(actor, crawler_tx.clone(), state.clone())
               , SoloAct);

    // database actor
    actor_builder.with_name(NAME_DB)
        .build(move |actor| actor::db_manager::run(actor, ai_tx.clone(), crawler_rx.clone()) 
               , SoloAct);

    // database actor
    actor_builder.with_name(NAME_AI)
        .build(move |actor| actor::ai_model::run(actor, ai_rx.clone())
               , SoloAct);
}

