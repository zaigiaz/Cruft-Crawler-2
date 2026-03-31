use std::env;
use std::path::PathBuf;



/// Setup context (e.g., at function start or main)
/// This is to reconstruct the correct path to wherever our executable is
pub struct FSContext {
    base_dir: PathBuf,
}

impl FSContext {
    // Enforce existence via `pushd` equivalent
    pub fn new(root: &str) -> std::io::Result<Self> {
        let current = env::current_dir()?;
        
        // Fail fast: ensure root exists.
        std::fs::create_dir_all(root)?;
        
        // Hard switch to root
        env::set_current_dir(root)?;
        
        Ok(FSContext {
            base_dir: PathBuf::from(root),
        })
    }
    
    // Append relative path
    pub fn path(&self, relative: &str) -> PathBuf {
        self.base_dir.join(relative)
    }
}
