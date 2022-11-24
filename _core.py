import os
import torch
import subprocess
import numpy as np
# import networkx as nx

from hashlib import sha256

GPU_PROG = "./BC.parallel.gpu.elf"
CPU_PROG = "./BC.parallel.cpu.elf"

def getTmpDir():
    for _tmp_dir in ["TMPDIR", "TMP", "TEMP"]:
        if _tmp_dir in os.environ.keys():
            return os.environ[_tmp_dir]
    return "/tmp/"


def turnCFG2Matrix(cfg: dict):
    nodes = list(cfg.keys())
    for k in cfg.keys():
        if isinstance(cfg[k], list):
            for e in cfg[k]:
                if e not in nodes:
                    nodes.append(e)
    node = [int(node) for node in nodes]
    size = max(node) + 1
    matrix = np.zeros((size, size), dtype=np.int32)
    for k in cfg.keys():
        if isinstance(cfg[k], list):
            for e in cfg[k]:
                matrix[int(k)][int(e)] = matrix[int(e)][int(k)] = 1
    return matrix


def getToken(cfg: dict):
    matrix = turnCFG2Matrix(cfg).tolist()
    return sha256(str(matrix).encode()).hexdigest()[:8]


def convertDictToGBCIN(cfg: dict, path: str):
    nodes = list(cfg.keys())
    lines = []
    for k in cfg.keys():
        for e in cfg[k]:
            lines.append((k, e))
            if e not in nodes:
                nodes.append(e)
    gbcin = ""
    gbcin += f"{len(nodes)} {len(lines)}\n"
    for node in nodes:
        gbcin += f"{node} "
    gbcin = gbcin[:-1] + '\n'
    for line in lines:
        gbcin += f"{line[0]} {line[1]} "
    gbcin = gbcin[:-1] + '\n'
    with open(path, 'w') as fgbcin:
        fgbcin.write(gbcin)
    return nodes, lines


def betweenness(cfg: dict, debug):

    token = getToken(cfg)
    output_path = f"{getTmpDir()}{token}.out"
    input_path = f"{getTmpDir()}{token}.in"

    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    nodes, lines = convertDictToGBCIN(cfg, input_path)

    if torch.cuda.is_available():
        PROG = GPU_PROG
    else:
        PROG = CPU_PROG

    command = [PROG, input_path, output_path]
    proc = subprocess.Popen(command)
    proc.wait()

    if not os.path.exists(output_path):
        raise AssertionError(f"[generation] output {output_path} failed")

    with open(output_path, "r") as fgbcout:
        fgbcout_content = fgbcout.read()
    bc = fgbcout_content.replace('\n', '').split(' ')[:-1]

    if len(bc) != len(nodes) + 1:
        raise AssertionError(f"[check] output {output_path} malformed ")

    max_bc = bc[0]
    bc = [float(bc_item) for bc_item in bc]
    sum_bc = sum(bc)
    addition = 1e-13
    bc = bc[1:]
    centralities = {}
    for i in range(len(nodes)):
        centralities[nodes[i]] = bc[i] / float(sum_bc + addition)

    if isinstance(debug, str) and debug == "DEADBEEF":
        print("debugging")
    else:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        # except Exception as e:
        #     print(f"{e} with \n\tinput {input_path}\n\toutput {output_path}")

    return centralities


if __name__ == "__main__":
    cfg = {
        '0': ['1', '116'], '1': ['2', '5', '11', '14', '17', '20', '23', '26', '30', '36', '43', '50', '55', '61', '64', '66', '70', '76', '84', '87', '93', '99', '106', '112', '117', '123', '126', '133', '137'], '2': ['3', '42'], '3': ['4', '165'], '4': ['33'], '5': ['6', '149'], '6': ['7', '176'], '7': ['8', '159'], '8': ['9', '10'], '9': ['10'], '10': ['143'], '11': ['12', '149'], '12': ['13', '176'], '13': ['96'], '14': ['15', '149'],
        '15': ['16', '176'], '16': ['96'], '17': ['18', '42'], '18': ['19', '165'], '19': ['33'], '20': ['21', '149'], '21': ['22', '91'], '22': ['90'], '23': ['24', '149'], '24': ['25', '176'], '25': ['96'], '26': ['27', '149'], '27': ['28', '176'], '28': ['29', '176'], '29': ['176'], '30': ['31', '42'], '31': ['32', '165'], '32': ['33'], '33': ['34', '164'], '34': ['35', '162'], '35': ['164'], '36': ['37', '42'], '37': ['38', '165'],
        '38': ['39', '165'], '39': ['40', '164'], '40': ['35', '41'], '41': ['163'], '42': ['165'], '43': ['44', '114'], '44': ['45', '115'], '45': ['46', '115'], '46': ['47', '115'], '47': ['48', '83'], '48': ['49', '82'], '49': ['83'], '50': ['51', '122'], '51': ['52', '171'], '52': ['53', '170'], '53': ['54', '121'], '54': ['167'], '55': ['56', '149'], '56': ['57', '176'], '57': ['58', '176'], '58': ['59', '131'], '59': ['60', '98'],
        '60': ['161'], '61': ['62', '149'], '62': ['63', '176'], '63': ['96'], '64': ['65', '132'], '65': ['128'], '66': ['67', '149'], '67': ['68', '176'], '68': ['69', '176'], '69': ['129'], '70': ['71', '150'], '71': ['72', '151'], '72': ['73', '151'], '73': ['74', '183'], '74': ['75', '181'], '75': ['183'], '76': ['77', '114'], '77': ['78', '115'], '78': ['79', '115'], '79': ['80', '115'], '80': ['81', '83'], '81': ['49', '82'],
        '82': ['83'], '83': ['115'], '84': ['85', '149'], '85': ['86', '176'], '86': ['96'], '87': ['88', '149'], '88': ['89', '91'], '89': ['90'], '90': ['91'], '91': ['92', '176'], '92': ['176'], '93': ['94', '149'], '94': ['95', '176'], '95': ['96'], '96': ['97', '131'], '97': ['98', '160'], '98': ['131'], '99': ['100', '149'], '100': ['101', '176'], '101': ['102', '176'], '102': ['103', '159'], '103': ['104', '105'], '104': ['105'],
        '105': ['144'], '106': ['107', '122'], '107': ['108', '171'], '108': ['109', '171'], '109': ['110', '170'], '110': ['111', '121'], '111': ['168'], '112': ['113', '132'], '113': ['125'], '114': ['115'], '115': ['165'], '116': ['176'], '117': ['118', '122'], '118': ['119', '171'], '119': ['120', '170'], '120': ['121', '167'], '121': ['170'], '122': ['171'], '123': ['124', '132'], '124': ['125'], '125': ['178'], '126': ['127', '132'],
        '127': ['128'], '128': ['129'], '129': ['130'], '130': ['131'], '131': ['176'], '132': ['178'], '133': ['134', '154'], '134': ['135', '155'], '135': ['136'], '136': ['147'], '137': ['138', '149'], '138': ['139', '176'], '139': ['140', '159'], '140': ['141', '142'], '141': ['142'], '142': ['143'], '143': ['144'], '144': ['145', '147'], '145': ['146', '180'], '146': ['147'], '147': ['148', '156'], '148': ['176'], '149': ['176'],
        '150': ['151'], '151': ['152', '158'], '152': ['153', '157'], '153': ['158'], '154': ['155'], '155': ['156', '176'], '156': ['176'], '157': ['158'], '158': ['166'], '159': ['176'], '160': ['161'], '161': ['130'], '162': ['163'], '163': ['164'], '164': ['165'], '165': ['166'], '166': ['176'], '167': ['168'], '168': ['169', '170'], '169': ['170'], '170': ['171'], '171': ['172', '175'], '172': ['173', '174'], '173': ['176'],
        '174': ['175'], '175': ['176'], '176': ['177', '178'], '177': ['178'], '178': ['179', '184'], '179': [], '180': ['136'], '181': ['182', '183'], '182': ['183'], '183': ['151'], '184': []
    }
    betweenness(cfg)
