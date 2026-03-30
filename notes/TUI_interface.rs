#![allow(unused)]

use steady_state::*;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use ratatui::{
    backend::CrosstermBackend,
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Line},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    DefaultTerminal,
};

// ── ACTOR LAYER (Unchanged except spawn_tui call) ────────────────────────────

pub async fn run(
    actor: SteadyActorShadow,
    ai_model_to_ui_rx: SteadyRx<String>,
    ui_to_db_tx: SteadyTx<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let actor = actor.into_spotlight([&ai_model_to_ui_rx], [&ui_to_db_tx]);
    if actor.use_internal_behavior {
        internal_behavior(actor, ai_model_to_ui_rx, ui_to_db_tx).await
    } else {
        actor.simulated_behavior(vec![&ai_model_to_ui_rx]).await
    }
}

async fn internal_behavior<A: SteadyActor>(
    mut actor: A,
    ai_model_to_ui_rx: SteadyRx<String>,
    ui_to_db_tx: SteadyTx<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut ai_model_to_ui_rx = ai_model_to_ui_rx.lock().await;
    let mut ui_to_db_tx = ui_to_db_tx.lock().await;

    // Channels between actor ↔ TUI
    let (suggest_tx, suggest_rx) = mpsc::channel::<PathBuf>();
    let (delete_tx, delete_rx) = mpsc::channel::<PathBuf>();

    // 🔥 SPAWN SEPARATED TUI — it handles its own lifecycle
    spawn_tui(suggest_rx, delete_tx);

    while actor.is_running(|| ai_model_to_ui_rx.is_closed_and_empty()) {
        // Forward AI verdicts → TUI
        while let Some(verdict) = actor.try_take(&mut ai_model_to_ui_rx) {
            if verdict.trim().to_lowercase() == "delete" {
                let _ = suggest_tx.send(PathBuf::from(&verdict));
            }
        }

        // Forward TUI actions → DB
        while let Ok(path) = delete_rx.try_recv() {
            actor.wait_vacant(&mut ui_to_db_tx, 1).await;
            actor.try_send(&mut ui_to_db_tx, path);
        }

        actor.wait_avail(&mut ai_model_to_ui_rx, 1).await;
    }

    Ok(())
}

// ── TUI MANAGER (NEW SEPARATED LAYER) ─────────────────────────────────────────

/// Manages the entire TUI lifecycle. Spawn once, it runs until quit.
pub fn spawn_tui(suggest_rx: mpsc::Receiver<PathBuf>, delete_tx: mpsc::Sender<PathBuf>) {
    thread::spawn(move || {
        let result = TuiManager::new(suggest_rx, delete_tx).run();
        if let Err(e) = result {
            eprintln!("TUI error: {}", e);
        }
    });
}

/// 🔥 COMPLETE TUI ENCAPSULATION
struct TuiManager {
    app: App,
    terminal: DefaultTerminal,
}

impl TuiManager {
    fn new(suggest_rx: mpsc::Receiver<PathBuf>, delete_tx: mpsc::Sender<PathBuf>) -> Self {
        let app = App::new(suggest_rx, delete_tx);
        let terminal = ratatui::init();
        Self { app, terminal }
    }

    /// Main TUI loop. Handles enter/exit automatically.
    fn run(mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.app.poll_suggestions(); // Initial poll
        
        loop {
            self.terminal.draw(|frame| self.render(frame))?;
            
            // Non-blocking poll for events + suggestions
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match key.code {
                        KeyCode::Char('q') => break, // Graceful exit
                        KeyCode::Up => self.app.move_up(),
                        KeyCode::Down => self.app.move_down(),
                        KeyCode::Char('d') => self.app.delete_selected(),
                        KeyCode::Char('k') => self.app.keep_selected(),
                        KeyCode::Char('n') => self.app.never_delete_selected(),
                        _ => {}
                    }
                }
            }
            
            self.app.poll_suggestions(); // Keep UI fresh
        }
        
        ratatui::restore(); // Cleanup
        Ok(())
    }
    
    fn render(&mut self, frame: &mut ratatui::Frame<CrosstermBackend<std::io::Stdout>>) {
        let vertical = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ]);
        let [list_area, hints_area, status_area] = vertical.areas(frame.area());

        // File list
        let items: Vec<ListItem> = self.app.suggested_files
            .iter()
            .enumerate()
            .map(|(i, path)| ListItem::new(format!("[{}] {}", i + 1, path.display())))
            .collect();

        let list = List::new(items)
            .block(Block::bordered().title(" CruftCrawler — Suggested Files "))
            .highlight_style(Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD))
            .highlight_symbol("▶ ");

        frame.render_stateful_widget(list, list_area, &mut self.app.list_state);

        // Hints
        let hints = Line::from(vec![
            " (↑↓) navigate ".into(),
            " (d) delete ".bold().fg(Color::Red),
            " (k) keep ".bold().fg(Color::Green),
            " (n) never-delete ".bold().fg(Color::Cyan),
            " (q) quit ".bold().fg(Color::Gray),
        ]);
        frame.render_widget(hints, hints_area);

        // Status
        let status = Paragraph::new(self.app.status.as_str()).fg(Color::DarkGray);
        frame.render_widget(status, status_area);
    }
}

// ── APP STATE (Moved inside TUI layer) ───────────────────────────────────────

struct App {
    suggested_files: Vec<PathBuf>,
    list_state: ListState,
    status: String,
    suggest_rx: mpsc::Receiver<PathBuf>,
    delete_tx: mpsc::Sender<PathBuf>,
}

impl App {
    fn new(suggest_rx: mpsc::Receiver<PathBuf>, delete_tx: mpsc::Sender<PathBuf>) -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        Self {
            suggested_files: Vec::new(),
            list_state,
            status: String::from("Waiting for AI suggestions..."),
            suggest_rx,
            delete_tx,
        }
    }

    // ... (keep all the existing methods: selected_path, clamp_selection, etc.)
    // [Methods omitted for brevity - copy from your original App impl]
}
