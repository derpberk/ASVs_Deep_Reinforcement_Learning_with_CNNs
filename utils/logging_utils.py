from rich.table import Table
from rich.console import Console
from rich.measure import Measurement
from rich.live import Live
from rich.align import Align

def present_hyperparameters(console, hyperparameters):

    assert isinstance(hyperparameters, dict), "Hyperparameters must be a dictionary"

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Hyperparameter Name", style="dim", width=25, justify="center")
    table.add_column("Value",justify="center",width=15)
    table.width = Measurement.get(console, table, console.width).maximum

    Align.center(table)

    for key in hyperparameters:

        table.add_row('[red]'+key+'[/red]', str(hyperparameters[key]))

    console.print(table)


def create_console():

    return Console()

def print_progress(console,progress):

    is_first = progress['episode_number'] == 1

    table = Table(show_header=is_first, header_style='bold #2070b2')

    table_centered = Align.center(table)

    with Live(table_centered, console=console, screen=False, refresh_per_second=20):

        table.add_column('Episode', justify='center')
        table.add_column('Acc. Reward', justify='center')
        table.add_column('Rol. Reward', justify='center')
        table.add_column('Record', justify='center')
        table.add_column('Epsilon', justify='center')
        table.add_column('Loss', justify='center')

        table.add_row("{: >5d}".format(progress['episode_number']),
                      "{:+.2f}".format(progress['reward']),
                      "{:+.2f}".format(progress['rolling_reward']),
                      "{:+.2f}".format(progress['record']),
                      "{:+.4f}".format(progress['epsilon']),
                      "{:+.3f}".format(progress['loss']))


