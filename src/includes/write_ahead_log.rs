use crate::includes::file_utils::File_Meta;

use std::fs::File;
use std::fs::OpenOptions;
use std::error::Error;
use std::io::Write;
use std::path::Path;

struct RotatingLog {
    path: String,
    max_size: u64,
}

impl RotatingLog {

    /// initialize and open the write_ahead_log
    fn new(path: &str, max_size: u64) -> Self {
        RotatingLog {
            path: path.to_string(),
            max_size,
        }
    }

    // /// write a batch to the file, this is to maintain data integrity with the database
    // fn write_batch(&self, batch: Vec<File_Meta>) -> std::io::Result<()> {
    // 	let file = OpenOptions::new()
    //         .read(true)
    //         .write(true)
    //         .create(true)
    //         .open(self.path);

    // 	while !batch.is_empty() {
    // 	    file.write_all
	    
    // 	}

    // }

    // /// rotate the log file (delete the last entry)
    // fn rotate_batch(&self) -> Result<Self, Box<dyn Error>> {
	
    // }

    // /// close the file handle
    // fn close_file(&self) -> Result<Self, Box<dyn Error>> {
	
    // }

}
