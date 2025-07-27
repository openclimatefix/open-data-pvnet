import typer
from open_data_pvnet.commands.gfs_samples import save_gfs_samples
# TODO: import any other commands here, e.g.
# from open_data_pvnet.commands.metoffice import metoffice_app

app = typer.Typer()

# register your new command
app.command("save-gfs-samples")(save_gfs_samples)  # ← this is your GFS‑sample saver

# register any other sub‑commands or apps
# app.add_typer(metoffice_app, name="metoffice")

def main():
    """Open‑Data‑PVNet command‑line interface."""
    app()

if __name__ == "__main__":
    main()
