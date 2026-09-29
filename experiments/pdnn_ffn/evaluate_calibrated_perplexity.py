"""Use the unchanged sliding-window evaluator with the typed checkpoint loader."""
import evaluate_multi_layer_full_path_perplexity as evaluator
from calibrated_pdnn_ffn import load_checkpoint

if __name__ == "__main__":
    evaluator.load_full_path_checkpoint = load_checkpoint
    evaluator.main()
