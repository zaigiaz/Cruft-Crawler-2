use crate::includes::file_utils::FileMetadata;
use std::io::{Seek, SeekFrom};
use std::fs::{OpenOptions, write};
use std::io::Write;

pub struct RotatingLog<'a> {
    pub path: &'a str,
    pub max_size: u64,
}

impl RotatingLog<'_> {
    
    /// write the last path that was crawled by the crawler iterator and save to log file
    pub fn write_log(&self, last_iter: &str) -> Result<(), std::io::Error> {

	let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(self.path)?;

	writeln!(file, "{}", last_iter).map_err(|e| {
            eprintln!("Couldn't write to file: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e)
	})?;

	Ok(())
    }


    /// check our WAL and then see if current batch is different from the last that was written to file
    /// if so then do all operations that arent in DB and update until we reach back to current data
    pub fn check_log(&self) -> Result<(), std::io::Error> {

	let mut file = OpenOptions::new()
	    .read(true)
	    .open(self.path)?;

	// seek end of file, then compare to end of DB
	file.seek(SeekFrom::Start(0))?;
	
	Ok(())
    }


    /// when file exceeds some max size, we clear it to save memory
    pub fn rotate_log(&mut self) -> Result<(), std::io::Error> {
	let metadata = std::fs::metadata(&self.path)?;

	// overwrite previous content of file with empty string
	if metadata.len() >= self.max_size {
	    std::fs::write(self.path, b"")?;
	}

	Ok(())
    }

}
