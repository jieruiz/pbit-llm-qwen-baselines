"""Reuse the established joint KD protocol with the AND-bank checkpoint type."""
from pathlib import Path
import sys
import train_joint_full_path_distillation as joint
from multithreshold_and_ffn import load_checkpoint


def load_training_student(path, device="cuda"):
    student,payload=load_checkpoint(path,device)
    # Same sampled bits and tied weights as the binary expansion. This training
    # factorization has tested identical real-arithmetic forward and STE gradients.
    student.binary_readout=False
    return student,payload


if __name__=="__main__":
    if "--output-dir" in sys.argv:
        output=Path(sys.argv[sys.argv.index("--output-dir")+1])
        if (output/"train_metrics.jsonl").exists():
            raise FileExistsError(f"Refusing to overwrite joint training: {output}")
    joint.load_full_path_checkpoint=load_training_student
    joint.main()
