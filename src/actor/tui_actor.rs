use steady_state::*;
use std::path::PathBuf;

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

    /// TODO :: Change this to actor channels, or remove because we dont need
    // let (suggest_tx, suggest_rx) = mpsc::channel::<PathBuf>();
    // let (delete_tx, delete_rx) = mpsc::channel::<PathBuf>();

    /// TODO :: Change this function to take different arguments
    spawn_tui(suggest_rx, delete_tx);

    /// TODO :: Change this to read delete_tree from DB
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
