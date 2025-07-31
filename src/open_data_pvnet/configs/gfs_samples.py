import argparse
from ocf_data_sampler import SamplerConfig, OCFDataSampler

def save_gfs_samples():
    """Convert GFS zarr files to .pt samples for PVNet training."""
    parser = argparse.ArgumentParser(
        description="Convert GFS zarr to .pt samples"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to GFS sampler config YAML"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Directory to save .pt samples"
    )
    args = parser.parse_args()

    cfg = SamplerConfig.from_yaml(args.config)
    sampler = OCFDataSampler(cfg)
    sampler.save_samples(args.output_dir)
    print(f"[INFO] Samples written to {args.output_dir}")

if __name__ == "__main__":
    save_gfs_samples()
