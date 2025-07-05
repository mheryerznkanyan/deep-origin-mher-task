#!/usr/bin/env python3
"""
Hydra-based training script for protein contact prediction model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import json
from pathlib import Path
import logging

# Set up logging
log = logging.getLogger(__name__)

class ConfigSaverCallback(Callback):
    """Callback to save the configuration used for training."""
    
    def __init__(self, config: DictConfig, save_path: str):
        super().__init__()
        self.config = config
        self.save_path = save_path
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save configuration at the start of training."""
        # Save full config
        with open(self.save_path, 'w') as f:
            OmegaConf.save(config=self.config, f=f)
        log.info(f"Configuration saved to {self.save_path}")



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
 
    # Create output directories
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Output directory: {output_dir}")
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    log.info(f"Log directory: {log_dir}")
    
    # Print configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Create data module
    log.info("Creating data module...")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Create model
    log.info("Creating model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # Create trainer
    log.info("Creating trainer...")
    
    # Prepare callbacks
    callbacks = []
    
    # Add config saver callback
    config_save_path = output_dir / "config.yaml"
    config_callback = ConfigSaverCallback(cfg, config_save_path)
    callbacks.append(config_callback)
    
    # Add other callbacks from config
    if hasattr(cfg.trainer, 'callbacks'):
        for callback_name, callback_config in cfg.trainer.callbacks.items():
            callback = hydra.utils.instantiate(callback_config)
            callbacks.append(callback)
    
    # Prepare loggers
    loggers = []
    if hasattr(cfg.trainer, 'logger'):
        for logger_name, logger_config in cfg.trainer.logger.items():
            logger = hydra.utils.instantiate(logger_config)
            loggers.append(logger)
    
    # Create trainer with callbacks and loggers
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_config.pop('callbacks', None)  # Remove callbacks from config
    trainer_config.pop('logger', None)     # Remove loggers from config
    trainer_config.pop('_target_', None)     # Remove loggers from config
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **trainer_config
    )
    
    # Train the model
    log.info("Starting training...")
    trainer.fit(model, datamodule)
    
    # Test the model
    log.info("Testing the model...")
    test_results = trainer.test(model, datamodule)
    
    # Save test results
    test_results_path = output_dir / "test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_results[0], f, indent=2)
    
    # Print test results
    log.info("Test Results:")
    for metric, value in test_results[0].items():
        log.info(f"{metric}: {value:.4f}")
    
    # Save experiment info
    experiment_info = {
        "experiment_name": cfg.experiment.name,
        "description": cfg.experiment.description,
        "tags": cfg.experiment.tags,
        "seed": cfg.seed,
        "output_dir": str(output_dir),
        "best_model_path": trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None,
        "test_results": test_results[0]
    }
    
    experiment_info_path = output_dir / "experiment_info.json"
    with open(experiment_info_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    log.info(f"Training completed! Results saved to: {output_dir}")
    log.info(f"Best model: {trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else 'Not available'}")

if __name__ == "__main__":
    main() 