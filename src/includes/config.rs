use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub ai_model: AIModel,
    pub crawler: Crawler,
}

#[derive(Debug, Deserialize)]
pub struct AIModel { pub model_path: String }

#[derive(Debug, Deserialize)]
pub struct Crawler { pub crawl_path: String }

pub fn read_toml(path_name: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let config_str = fs::read_to_string(path_name)
        .map_err(|e| format!("Failed to read '{}': {}", path_name, e))?;
    let config: Config = toml::from_str(&config_str)
        .map_err(|e| format!("Failed to parse TOML in '{}': {}", path_name, e))?;
    Ok(config)
}
