import csv
import logging
import os
import sys
import pathlib
import pickle
import shutil
import traceback
import json

import click
import hyperactive
import jesse.helpers as jh
import numpy as np
import pandas as pd
import pkg_resources
import yaml
from jesse.research import backtest, get_candles, import_candles
from .candledates import get_first_and_last_date
from time import sleep


def validate_cwd() -> None:
    """
    make sure we're in a Jesse project
    """
    ls = os.listdir('.')
    is_jesse_project = 'strategies' in ls and 'storage' in ls

    if not is_jesse_project:
        print('Current directory is not a Jesse project. You must run commands from the root of a Jesse project.')
        exit()


validate_cwd()

logger = logging.getLogger()
logger.addHandler(logging.FileHandler("jesse-hyperactive.log", mode="w"))

empty_backtest_data = {'total': 0, 'total_winning_trades': None, 'total_losing_trades': None,
                       'starting_balance': None, 'finishing_balance': None, 'win_rate': None,
                       'ratio_avg_win_loss': None, 'longs_count': None, 'longs_percentage': None,
                       'shorts_percentage': None, 'shorts_count': None, 'fee': None, 'net_profit': None,
                       'net_profit_percentage': None, 'average_win': None, 'average_loss': None, 'expectancy': None,
                       'expectancy_percentage': None, 'expected_net_profit_every_100_trades': None,
                       'average_holding_period': None, 'average_winning_holding_period': None,
                       'average_losing_holding_period': None, 'gross_profit': None, 'gross_loss': None,
                       'max_drawdown': None, 'annual_return': None, 'sharpe_ratio': None, 'calmar_ratio': None,
                       'sortino_ratio': None, 'omega_ratio': None, 'serenity_index': None, 'smart_sharpe': None,
                       'smart_sortino': None, 'total_open_trades': None, 'open_pl': None, 'winning_streak': None,
                       'losing_streak': None, 'largest_losing_trade': None, 'largest_winning_trade': None,
                       'current_streak': None}


# create a Click group
@click.group()
@click.version_option(pkg_resources.get_distribution("jesse-hyperactive").version)
def cli() -> None:
    pass


@cli.command()
def create_config() -> None:
    target_dirname = pathlib.Path().resolve()
    package_dir = pathlib.Path(__file__).resolve().parent
    shutil.copy2(f'{package_dir}/hyperactive-config.yml', f'{target_dirname}/hyperactive-config.yml')


@cli.command()
def run()->None:
    cfg = get_config()
    update_config(cfg)
    run_optimization()


def run_optimization(batchmode=False, cfg=None, hp_dict=None) -> None:
    if cfg == None: # load the cfg if it wasnt handed over
        cfg = get_config()

    # tell the user which symbol is optimized

    print("Run Study for ", cfg['symbol'], " from date: ", cfg['timespan']['start_date'])
    study_name = get_study_name(cfg)

    path = get_csv_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hp_dict is None:
        StrategyClass = jh.get_strategy_class(cfg['strategy_name'])
        hp_dict = StrategyClass().hyperparameters()

    search_space = get_search_space(hp_dict)

    # Later use actual search space combinations to determin n_iter / population size?
    #combinations_count = 1
    #for value in search_space.values():
        #combinations_count *= len(value)

    mem = None

    if jh.file_exists(path):
        with open(path, "r") as f:
            mem = pd.read_csv(f, sep="\t", na_values='nan')
        if not mem.empty and not click.confirm(
                f'Previous optimization results for {study_name} exists. Continue?',
                default=True,
        ):
            mem = None

    hyper = hyperactive.Hyperactive(distribution="joblib")

        # Evolution Strategy    
    if cfg['optimizer'] == "EvolutionStrategyOptimizer":
        optimizer = hyperactive.optimizers.EvolutionStrategyOptimizer(
            population=cfg[cfg['optimizer']]['population'],
            mutation_rate=cfg[cfg['optimizer']]['mutation_rate'],
            crossover_rate=cfg[cfg['optimizer']]['crossover_rate'],
            rand_rest_p=cfg[cfg['optimizer']]['rand_rest_p'],
        )

        if mem is None or len(mem) < cfg[cfg['optimizer']]['population']:
            if mem is not None and len(mem) < cfg[cfg['optimizer']]['population']:
                print('Previous optimization has too few individuals for population. Reinitialization necessary.')
            # init empty pandas dataframe
            search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
            with open(path, "w") as f:
                search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,
                             initialize={"random": cfg[cfg['optimizer']]['population']},
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])
        else:
            mem.drop([f'training_{k}' for k in empty_backtest_data.keys()] + [f'testing_{k}' for k in
                                                                              empty_backtest_data.keys()], 1,
                     inplace=True)
            hyper.add_search(objective, search_space, optimizer=optimizer, memory_warm_start=mem,
                             n_iter=cfg['n_iter'] -  len(mem),
                             n_jobs=cfg['n_jobs'])  

    # Hill Climbing
    elif cfg['optimizer'] == "HillClimbingOptimizer":
        optimizer = hyperactive.optimizers.HillClimbingOptimizer(
            epsilon=cfg[cfg['optimizer']]['epsilon'],
            distribution=cfg[cfg['optimizer']]['distribution'],
            n_neighbours=cfg[cfg['optimizer']]['n_neighbours'],
            rand_rest_p=cfg[cfg['optimizer']]['rand_rest_p'],            
        )       
        # init empty pandas dataframe
        search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
        with open(path, "w") as f:
            search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,          
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])                         

    # Random Restart Hill Climbing
    elif cfg['optimizer'] == "RandomRestartHillClimbingOptimizer":
        optimizer = hyperactive.optimizers.RandomRestartHillClimbingOptimizer(
            epsilon=cfg[cfg['optimizer']]['epsilon'],
            distribution=cfg[cfg['optimizer']]['distribution'],
            n_neighbours=cfg[cfg['optimizer']]['n_neighbours'],
            rand_rest_p=cfg[cfg['optimizer']]['rand_rest_p'],
            n_iter_restart=cfg[cfg['optimizer']]['n_iter_restart'],
        )       
        # init empty pandas dataframe
        search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
        with open(path, "w") as f:
            search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,          
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])
    
    # Powells Method
    elif cfg['optimizer'] == "PowellsMethod":
        optimizer = hyperactive.optimizers.PowellsMethod(
            iters_p_dim=cfg[cfg['optimizer']]['iters_p_dim'],          
        )       
        # init empty pandas dataframe
        search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
        with open(path, "w") as f:
            search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,          
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])

    # Particle Swarm Optimization
    elif cfg['optimizer'] == "ParticleSwarmOptimizer":
        optimizer = hyperactive.optimizers.ParticleSwarmOptimizer(
            inertia=cfg[cfg['optimizer']]['inertia'],
            cognitive_weight=cfg[cfg['optimizer']]['cognitive_weight'],
            social_weight=cfg[cfg['optimizer']]['social_weight'],
            temp_weight=cfg[cfg['optimizer']]['temp_weight'],
            rand_rest_p=cfg[cfg['optimizer']]['rand_rest_p'],
        )       
        # init empty pandas dataframe
        search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
        with open(path, "w") as f:
            search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,          
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])

    # TreeStructuredParzenEstimators
    elif cfg['optimizer'] == "TreeStructuredParzenEstimators":
        optimizer = hyperactive.optimizers.TreeStructuredParzenEstimators(
            gamma_tpe=cfg[cfg['optimizer']]['gamma_tpe'],
            rand_rest_p=cfg[cfg['optimizer']]['rand_rest_p'],         
        )       
        # init empty pandas dataframe
        search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
        with open(path, "w") as f:
            search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,          
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])

    # RandomSearch
    elif cfg['optimizer'] == "RandomSearchOptimizer":
        optimizer = hyperactive.optimizers.RandomSearchOptimizer()       
        # init empty pandas dataframe
        search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'{k}' for k in empty_backtest_data.keys()])
        with open(path, "w") as f:
            search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,          
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs']) 

    else:
        raise ValueError(f'Entered optimizer which is {cfg["optimizer"]} is not known.')

    hyper.run()

@cli.command()
def batchrun() -> None:
    cfg = get_config()
    batch_dict = get_batch_dict()
    print("Going to run the optimization for the symbols: ", batch_dict["symbols"])
    start_date_dict = import_candles_batch(batch_dict=batch_dict, cfg=cfg)

    # step1 Optimization
    # iterate over the symbols
    for symbol in batch_dict['symbols']:
        # run the optimzation for every hyperparameterset
        for hp_set in batch_dict['search_hyperparamerts']:
            #print(hp_set, batch_dict['search_hyperparamerts'][hp_set])
            #setup the run config
            cfg['timespan']['start_date'] = start_date_dict[symbol]
            cfg['symbol'] = symbol
            cfg['hp_set'] = hp_set
            update_config(cfg)
            run_optimization(batchmode=True, cfg=cfg, hp_dict=batch_dict['search_hyperparamerts'][hp_set])


def get_batch_dict() -> dict:
    batch_path = "hyperactive_batch.json"
    batch_path = os.path.abspath(batch_path)
    if not os.path.isfile(batch_path):
        print("There is no file with symbols which should be optimized.")
        sleep(0.5)
        batch_dict = {
                    "symbols": ["BTC-USDT", "ETH-USDT"],
                    "search_hyperparamerts":{
                            "safe":[
                                    {
                                    "name": "sma",
                                    "type": "int",
                                    "min": 20,
                                    "max": 30,
                                    "default": 25
                                    }
                                ],
                            "risky":[
                                    {
                                    "name": "sma",
                                    "type": "int",
                                    "min": 2,
                                    "max": 5,
                                    "default": 2
                                    }
                                ]
                            }
                    }
        with open(batch_path, 'w') as outfile:
            json.dump(batch_dict, outfile, indent=4, sort_keys=True)
        print("I created a file for you at ", batch_path , ":)")
        sleep(0.5)
        print("Please fill in your symbols and restart with: 'jesse-optuna batchrun' again")
        sleep(0.5)
        exit()
    else:
        try:
            with open(batch_path, 'r', encoding='UTF-8') as dna_settings: 
                batch_dict = json.load(dna_settings)
                if "search_hyperparamerts" in batch_dict:
                    for hp_dicts in batch_dict["search_hyperparamerts"]:
                        print(hp_dicts, batch_dict["search_hyperparamerts"][hp_dicts])
                        for hp in batch_dict["search_hyperparamerts"][hp_dicts]: 
                            print("hp", batch_dict["search_hyperparamerts"][hp_dicts], hp["type"])
                            hp["type"] = eval(hp["type"])

                return batch_dict
        except json.JSONDecodeError: 
            raise (
            'DNA Settings file is formatted wrong.'
            )
        except:
            raise

def import_candles_batch(batch_dict, cfg) -> dict:
    # get optimzing dates from the config
    timespans = []
    for key in cfg: 
        if 'timespan' in key: 
            timespans.append({
                            'start_date': cfg[key]['start_date'],
                            'finish_date': cfg[key]['finish_date']
                            })
    # get the first date so we can import all needed candles 
    first_date = 0
    for i, date_i in enumerate(timespans[1:]):
        if jh.date_to_timestamp(date_i['start_date']) < jh.date_to_timestamp(timespans[first_date]['start_date']):
            first_date = i+1

    print(f"First testing date is {timespans[first_date]['start_date']}.")

    for symbol in batch_dict['symbols']: 
        import_candles(cfg['exchange'], str(symbol), timespans[first_date]['start_date'], True)

    # check if candles are imported succesfully for all symbols: 
    start_date_dict = {}
    for i, symbol in enumerate(batch_dict["symbols"]):
        print("checking if all needed candles are imported for symbol {}".format(symbol))
        succes, start_date, finish_date, message = get_first_and_last_date(cfg['exchange'], str(symbol), timespans[first_date]['start_date'], timespans[first_date]['finish_date'])
        if not succes: 
            if start_date is None:
                print(message)
                exit()
            if not message is None: # if first backtestable timestamp is in the future, that means we have some but not enough candles
                print("Not Enough candles!")
                print(message)
                exit()
            else:
                print("First available date is {}".format(start_date))
                print("Changing the start date for this symbol")
                start_date_dict[symbol] = start_date
                continue

        start_date_dict[symbol] = timespans[first_date]['start_date']
    return start_date_dict

def get_config(running=False):
    if running: 
        cfg_file = pathlib.Path('.run-hyperactive-config.yml')
    else:
        cfg_file = pathlib.Path('hyperactive-config.yml')

    if not cfg_file.is_file():
        print("hyperactive-config.yml not found. Run create-config command.")
        exit()
    else:
        with open(cfg_file, "r") as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)

    return cfg

def update_config(cfg) -> None: 
    cfg_file = pathlib.Path('.run-hyperactive-config.yml')
    with open(cfg_file, "w") as ymlfile:
        yaml.safe_dump(cfg, ymlfile)

def create_run_config(cfg, run_settings) -> dict:
    # create a copy of the dict
    run_config = dict(cfg)
    for key in run_settings:
        run_config[key] = run_settings[key]
    return run_config


def get_study_name(cfg) -> str:
    if 'hp_set' in cfg:
        hp_set = f"-{cfg['hp_set']}"
    else:
        hp_set = ''
    return f"{cfg['strategy_name']}-{cfg['exchange']}-{cfg['optimizer']}-{cfg['symbol']}-{cfg['timeframe']}{hp_set}"

def get_csv_path(cfg) -> str:
    study_name = get_study_name(cfg)
    if 'hp_set' in cfg:
        hp_set = f"/{cfg['hp_set']}"
    else:
        hp_set = ''
    return f"storage/jesse-hyperactive/csv/{cfg['symbol']}{hp_set}/{study_name}.csv"

def objective(opt):
    cfg = get_config(running=True)

    try:
        training_data_metrics = backtest_function(cfg['timespan']['start_date'],
                                                  cfg['timespan']['finish_date'],
                                                  hp=opt, cfg=cfg)
    except Exception as err:
        logger.error("".join(traceback.TracebackException.from_exception(err).format()))
        return np.nan

    if training_data_metrics is None:
        return np.nan

    if training_data_metrics['total'] <= 5:
        return np.nan

    total_effect_rate = np.log10(training_data_metrics['total']) / np.log10(cfg['optimal-total'])
    total_effect_rate = min(total_effect_rate, 1)
    ratio_config = cfg['fitness-ratio']
    if ratio_config == 'sharpe':
        ratio = training_data_metrics['sharpe_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'calmar':
        ratio = training_data_metrics['calmar_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 30)
    elif ratio_config == 'sortino':
        ratio = training_data_metrics['sortino_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    elif ratio_config == 'omega':
        ratio = training_data_metrics['omega_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'serenity':
        ratio = training_data_metrics['serenity_index']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    elif ratio_config == 'smart sharpe':
        ratio = training_data_metrics['smart_sharpe']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'smart sortino':
        ratio = training_data_metrics['smart_sortino']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    else:
        raise ValueError(
            f'The entered ratio configuration `{ratio_config}` for the optimization is unknown. Choose between sharpe, calmar, sortino, serenity, smart shapre, smart sortino and omega.')
    if ratio < 0:
        return np.nan

    score = total_effect_rate * ratio_normalized

    # you can access the entire dictionary from "para"
    parameter_dict = opt.para_dict

    # save the score in the copy of the dictionary
    parameter_dict["score"] = score

    for key, value in training_data_metrics.items():
        parameter_dict[f'training_{key}'] = value

    path = get_csv_path(cfg)

    # append parameter dictionary to csv
    with open(path, "a") as f:
        writer = csv.writer(f, delimiter='\t')
        fields = parameter_dict.values()
        writer.writerow(fields)

    return score


def get_search_space(strategy_hps):
    hp = {}
    for st_hp in strategy_hps:
        if st_hp['type'] is int:
            if 'step' not in st_hp:
                st_hp['step'] = 1
            hp[st_hp['name']] = list(range(st_hp['min'], st_hp['max'] + st_hp['step'], st_hp['step']))
        elif st_hp['type'] is float:
            if 'step' not in st_hp:
                st_hp['step'] = 0.1
            decs = str(st_hp['step'])[::-1].find('.')
            hp[st_hp['name']] = list(
                np.trunc(np.arange(st_hp['min'], st_hp['max'] + st_hp['step'], st_hp['step']) * 10 ** decs) / (
                        10 ** decs))
        elif st_hp['type'] is bool:
            hp[st_hp['name']] = [True, False]
        else:
            raise TypeError('Only int, bool and float types are implemented')
    return hp


def get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str) -> np.ndarray:
    path = pathlib.Path('storage/jesse-hyperactive')
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pickle"
    cache_file = pathlib.Path(f'storage/jesse-hyperactive/{cache_file_name}')

    if cache_file.is_file():
        with open(f'storage/jesse-hyperactive/{cache_file_name}', 'rb') as handle:
            candles = pickle.load(handle)
    else:
        candles = get_candles(exchange, symbol, '1m', start_date, finish_date)
        with open(f'storage/jesse-hyperactive/{cache_file_name}', 'wb') as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return candles


def backtest_function(start_date, finish_date, hp, cfg):
    candles = {}
    extra_routes = []
    if cfg['extra_routes'] is not None:
        for extra_route in cfg['extra_routes'].items():
            extra_route = extra_route[1]
            candles[jh.key(extra_route['exchange'], extra_route['symbol'])] = {
                'exchange': extra_route['exchange'],
                'symbol': extra_route['symbol'],
                'candles': get_candles_with_cache(
                    extra_route['exchange'],
                    extra_route['symbol'],
                    start_date,
                    finish_date,
                ),
            }
            extra_routes.append({'exchange': extra_route['exchange'], 'symbol': extra_route['symbol'],
                                 'timeframe': extra_route['timeframe']})
    candles[jh.key(cfg['exchange'], cfg['symbol'])] = {
        'exchange': cfg['exchange'],
        'symbol': cfg['symbol'],
        'candles': get_candles_with_cache(
            cfg['exchange'],
            cfg['symbol'],
            start_date,
            finish_date,
        ),
    }

    route = [{'exchange': cfg['exchange'], 'strategy': cfg['strategy_name'], 'symbol': cfg['symbol'],
              'timeframe': cfg['timeframe']}]

    config = {
        'starting_balance': cfg['starting_balance'],
        'fee': cfg['fee'],
        'futures_leverage': cfg['futures_leverage'],
        'futures_leverage_mode': cfg['futures_leverage_mode'],
        'exchange': cfg['exchange'],
        'settlement_currency': cfg['settlement_currency'],
        'warm_up_candles': cfg['warm_up_candles']
    }

    backtest_data = backtest(config, route, extra_routes=extra_routes, candles=candles, hyperparameters=hp)['metrics']

    if backtest_data['total'] == 0:
        backtest_data = empty_backtest_data

    return backtest_data
