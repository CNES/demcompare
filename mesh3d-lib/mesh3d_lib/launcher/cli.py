"""
CLI commands
"""

import click

from .runner import run
from .scripts import denoise_run, filter_run, mesh_run, texture_run, main_run


@click.group()
def cli():
    """Welcome to the CNES R&T Mesh 3D main CLI."""
    pass


@cli.command()
@click.argument('configuration')
def denoising(configuration):
    """Train a model according to a CONFIGURATION file."""
    run(denoise_run, config_filepath=configuration)


@cli.command()
@click.argument('configuration')
def filtering(configuration):
    """Infer with a model according to a CONFIGURATION file."""
    run(filter_run, config_filepath=configuration)


@cli.command()
@click.argument('configuration')
def meshing(configuration):
    """Evaluate with a model according to a CONFIGURATION file."""
    run(mesh_run, config_filepath=configuration)


@cli.command()
@click.argument('configuration')
def texturing(configuration):
    """Evaluate with a model according to a CONFIGURATION file."""
    run(texture_run, config_filepath=configuration)


@cli.command()
@click.argument('configuration')
def pipeline(configuration):
    """Evaluate with a model according to a CONFIGURATION file."""
    run(main_run, config_filepath=configuration)


if __name__ == '__main__':
    cli()
