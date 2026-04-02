use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(rename = "ai_model")]
    pub ai_model: AIModel,
    
    #[serde(rename = "crawler")]
    pub crawler: Crawler,
}

#[derive(Debug, Deserialize)]
pub struct AIModel {
    pub model_path: String,
}

#[derive(Debug, Deserialize)]
pub struct Crawler {
    pub crawl_path: String,
}

/// read toml file as a config file for our program
pub fn read_toml() -> Result<(Config), Box<dyn std::error::Error>> {

    // Load config.toml from project root
    let config_str = fs::read_to_string("../../config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    Ok(config)
}
