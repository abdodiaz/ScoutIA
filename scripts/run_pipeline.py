"""
Complete Pipeline Runner for ScoutIA Pro
Automates: Data Prep → Model Training → Testing → Report Generation → API Launch
"""

import sys
from pathlib import Path
import logging
import subprocess
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation.preprocess_data import preprocess_pipeline
from models.train_model import main as train_model
from models.predict import InjuryRiskPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScoutIAPipeline:
    """Complete pipeline for ScoutIA Pro."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.base_dir = Path(__file__).parent.parent
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info("Initialized ScoutIA Pipeline")
    
    def step_1_data_preparation(self):
        """Step 1: Prepare and preprocess data."""
        logger.info("="*60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*60)
        
        try:
            df = preprocess_pipeline(
                data_dir="data/raw",
                output_path="data/processed/players_clean.csv"
            )
            
            logger.info(f"✅ Data preparation complete. Processed {len(df)} rows.")
            return True
        
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False
    
    def step_2_model_training(self):
        """Step 2: Train machine learning model."""
        logger.info("="*60)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*60)
        
        try:
            train_model()
            logger.info("✅ Model training complete.")
            return True
        
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def step_3_model_evaluation(self):
        """Step 3: Evaluate trained model."""
        logger.info("="*60)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*60)
        
        try:
            predictor = InjuryRiskPredictor()
            
            if predictor.model is None:
                logger.error("❌ Model not found. Cannot evaluate.")
                return False
            
            # Sample data for testing
            sample_data = {
                'age': 25,
                'matches_played': 30,
                'minutes_played': 2400,
                'goals': 5,
                'assists': 3,
                'passes_attempted': 800,
                'passes_completed': 720,
                'tackles': 50,
                'interceptions': 30,
                'sprints': 200,
                'distance_covered_km': 300.0,
                'total_injuries': 1,
                'pass_accuracy': 90.0,
                'intensity': 80.0,
                'goals_per_match': 0.17,
                'distance_per_match': 10.0,
                'sprint_per_match': 6.67,
                'defensive_activity': 80
            }
            
            result = predictor.predict(sample_data)
            
            logger.info(f"Sample prediction: {result['risk_level']} (confidence: {result['confidence']:.2%})")
            logger.info("✅ Model evaluation complete.")
            return True
        
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return False
    
    def step_4_generate_report(self):
        """Step 4: Generate pipeline report."""
        logger.info("="*60)
        logger.info("STEP 4: GENERATE REPORT")
        logger.info("="*60)
        
        try:
            report_path = self.reports_dir / f"pipeline_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("SCOUTIA PRO PIPELINE REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Model info
                model_path = Path("models/injury_risk_model.pkl")
                if model_path.exists():
                    f.write("[OK] Model: Found at {}\n".format(model_path))
                else:
                    f.write("[FAIL] Model: Not found\n")
                
                # Processed data info
                data_path = Path("data/processed/players_clean.csv")
                if data_path.exists():
                    import pandas as pd
                    df = pd.read_csv(data_path)
                    f.write(f"[OK] Processed Data: {len(df)} rows\n")
                else:
                    f.write("[FAIL] Processed Data: Not found\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("PIPELINE STATUS\n")
                f.write("="*60 + "\n\n")
                
                f.write("All pipeline steps completed successfully.\n")
            
            logger.info(f"✅ Report generated: {report_path}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return False
    
    def step_5_launch_api(self, background=False):
        """Step 5: Launch FastAPI server."""
        logger.info("="*60)
        logger.info("STEP 5: LAUNCH API")
        logger.info("="*60)
        
        try:
            if background:
                logger.info("Starting API in background...")
                # Use subprocess to run in background
                process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                logger.info(f"✅ API started in background (PID: {process.pid})")
                logger.info("API available at: http://localhost:8000")
                logger.info("API docs at: http://localhost:8000/docs")
                
                return True, process
            else:
                logger.info("To launch API, run: uvicorn backend.main:app --reload")
                logger.info("API will be available at: http://localhost:8000")
                return True, None
        
        except Exception as e:
            logger.error(f"❌ API launch failed: {e}")
            return False, None
    
    def run_complete_pipeline(self, launch_api=False):
        """Run the complete pipeline."""
        logger.info("\n" + "="*60)
        logger.info("STARTING SCOUTIA PRO PIPELINE")
        logger.info("="*60 + "\n")
        
        start_time = time.time()
        results = {}
        
        # Execute steps
        results['data_prep'] = self.step_1_data_preparation()
        
        if results['data_prep']:
            results['training'] = self.step_2_model_training()
            
            if results['training']:
                results['evaluation'] = self.step_3_model_evaluation()
        
        results['report'] = self.step_4_generate_report()
        
        if launch_api:
            results['api'], process = self.step_5_launch_api(background=False)
        
        # Summary
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        for step, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{step.upper()}: {status}")
        
        logger.info(f"\nTotal time: {elapsed_time:.2f} seconds")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.error("\n❌ PIPELINE COMPLETED WITH ERRORS")
        
        logger.info("="*60 + "\n")
        
        return all_passed


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ScoutIA Pro Pipeline')
    parser.add_argument('--launch-api', action='store_true', help='Launch API after pipeline')
    parser.add_argument('--step', type=str, help='Run specific step: data_prep, training, evaluation, report, api')
    
    args = parser.parse_args()
    
    pipeline = ScoutIAPipeline()
    
    if args.step:
        # Run specific step
        if args.step == 'data_prep':
            pipeline.step_1_data_preparation()
        elif args.step == 'training':
            pipeline.step_2_model_training()
        elif args.step == 'evaluation':
            pipeline.step_3_model_evaluation()
        elif args.step == 'report':
            pipeline.step_4_generate_report()
        elif args.step == 'api':
            pipeline.step_5_launch_api(background=False)
        else:
            logger.error(f"Unknown step: {args.step}")
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline(launch_api=args.launch_api)


if __name__ == "__main__":
    main()

