use crate::includes::file_utils::File_Meta;
use std::io::{Seek, SeekFrom};
use std::fs::OpenOptions;
use std::io::Write;

struct RotatingLog {
    path: String,
    max_size: u64,
}

impl RotatingLog {
    
    /// write the last path that was crawled by the crawler iterator and save to log file
    pub fn write_log(WriteFile: &str, last_iter: &str) -> Result<(), std::io::Error> {

	let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(WriteFile)?;

	writeln!(file, "{}", last_iter).map_err(|e| {
            eprintln!("Couldn't write to file: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e)
	})?;

	Ok(())
    }


    /// check our WAL and then see if current batch is different from the last that was written to file
    /// if so then do all operations that arent in DB and update until we reach back to current data
    pub fn check_log(Readfile: &str) -> Result<(), std::io::Error> {

	let mut file = OpenOptions::new()
	    .read(true)
	    .open(Readfile)?;

	// seek end of file, then compare to end of DB
	file.seek(SeekFrom::Start(0))?;
	
	Ok(())
    }


    
    pub fn rotate_log() -> Result<(), std::io::Error> {
	println!("delete the log after 10kb here");

	Ok(())
    }

}
