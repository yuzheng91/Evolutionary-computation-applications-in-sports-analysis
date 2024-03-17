import numpy as np
import pandas as pd


def get_feature():
    train_label = np.array(
        [
            0.567,
            0.487,
            0.655,
            0.431,
            0.452,
            0.306,
            0.570,
            0.622,
            0.278,
            0.527,
            0.236,
            0.473,
            0.626,
            0.570,
            0.519,
            0.526,
            0.653,
            0.319,
            0.431,
            0.546,
            0.306,
            0.292,
            0.667,
            0.692,
            0.564,
            0.431,
            0.452,
            0.375,
            0.699,
            0.456,
            0.517,
            0.613,
            0.517,
            0.540,
            0.518,
            0.524,
            0.610,
            0.563,
            0.281,
            0.664,
            0.244,
            0.305,
            0.500,
            0.402,
            0.660,
            0.640,
            0.617,
            0.551,
            0.444,
            0.451,
            0.293,
            0.268,
            0.606,
            0.747,
            0.329,
            0.366,
            0.410,
            0.568,
            0.580,
            0.427,
            0.494,
            0.667,
            0.523,
            0.488,
            0.329,
            0.598,
            0.463,
            0.677,
            0.207,
            0.526,
            0.268,
            0.427,
            0.517,
            0.525,
            0.602,
            0.542,
            0.678,
            0.494,
            0.506,
            0.570,
            0.488,
            0.415,
            0.656,
            0.548,
            0.402,
            0.573,
            0.268,
            0.494,
            0.451,
            0.427,
        ]
    )

    columns = [
        "Team",
        "Player Name",
        "MP",
        "PTS",
        "AST",
        "ORB",
        "DRB",
        "BLK",
        "STL",
        "TOV",
        "PF",
        "2PA",
        "3PA",
        "FTA",
        "eFG%",
        "2P%",
        "3P%",
        "FT%",
        "PER",
        "WS/48",
    ]

    Team_List = {}
    files = ["2020-2021.csv", "2021-2022.csv", "2022-2023.csv"]
    for filename in files:
        df = pd.read_csv(f"dataset/{filename}", usecols=columns)

        # Get all players in the dataset
        Player_List = []
        for i in range(len(list(df["Player Name"]))):
            states = {}
            for str in columns:
                states[str] = list(df[str])[i]
            Player_List.append(states)

        # Push each player into his own team
        for i in range(len(Player_List)):
            team_name = Player_List[i]["Team"][:9]
            if team_name + filename[:4] not in Team_List:
                Team_List[team_name + filename[:4]] = []

            Team_List[team_name + filename[:4]].append(Player_List[i])

    # Each team is a training data
    feature_name = columns[3:].copy()
    num_team = len(Team_List.keys())
    num_feat = len(feature_name)
    train_data = np.zeros((num_team, num_feat))

    """Scale total minutes to 240 for all team states"""
    team_time = []
    for team_name, player_states in Team_List.items():
        time = 0.0
        for states in player_states:
            time += states["MP"]
        team_time.append(time)

    k = 0
    for team_name, player_states in Team_List.items():
        # print(team_name)
        # print(player_states)
        for states in player_states:
            mp = (states["MP"] * (240.0 / team_time[k])) / 48
            for i in range(num_feat):
                feat = feature_name[i]
                train_data[k][i] += mp * (states[feat] * (240.0 / team_time[k]))

        k += 1

    """Delete Trash Team"""
    winrate_thres = 0.35
    indice = []
    for i in range(len(train_label)):
        if train_label[i] > winrate_thres:
            indice.append(i)

    train_data = train_data[indice]
    train_label = train_label[indice]

    return train_data, train_label, num_feat


def LAC_feature():
    columns = [
        "Team",
        "Player Name",
        "MP",
        "PTS",
        "AST",
        "ORB",
        "DRB",
        "BLK",
        "STL",
        "TOV",
        "PF",
        "2PA",
        "3PA",
        "FTA",
        "eFG%",
        "2P%",
        "3P%",
        "FT%",
        "PER",
        "WS/48",
    ]

    df = pd.read_csv(f"dataset/LAC.csv", usecols=columns)

    # Get all players in the dataset
    Player_List = []
    for i in range(len(list(df["Player Name"]))):
        states = {}
        for str in columns:
            states[str] = list(df[str])[i]
        Player_List.append(states)

    feature_name = columns[3:].copy()
    num_feat = len(feature_name)
    num_player = len(Player_List)
    test_data = np.zeros((num_player, num_feat))

    for i in range(num_player):
        for j in range(num_feat):
            feat = feature_name[j]
            test_data[i][j] = Player_List[i][feat] * (240.0 / 266.0)

    time = [state["MP"] * (240.0 / 266.0) for state in Player_List]
    return test_data, time, num_feat
