package ar.uba.fi.celdas;

import tracks.ArcadeMachine;

import java.util.Random;


public class TestCeldas {

    public static void main(String[] args) {
        //Available controllers:
        String celdasAgent = "ar.uba.fi.celdas.Agent";

        //Available games:
        String gamesPath = "examples/gridphysics/";
        String[] games = new String[]{};
        String generateLevelPath = "examples/gridphysics/";

        //Other settings
        boolean visuals = true;
        int seed = new Random().nextInt();

        //Game and level to play
        int gameIdx = 44;
        int levelIdx = 0; //level names from 0 to 4 (game_lvlN.txt).
        String game = gamesPath + "labyrinth" + ".txt";
        // String level1 = gamesPath + games[gameIdx] + "_lvl" +levelIdx +".txt";
        String level1 = gamesPath + "labyrinth_lvl_easy_" + levelIdx + ".txt";

        String recordLevelFile = generateLevelPath + "labyrinth_glvl.txt";
        String recordActionsFile = null;//"actions_" + games[gameIdx] + "_lvl" + levelIdx + "_" + seed + ".txt"; //where to record the actions executed. null if not to save.

        // 1. This starts a game, in a level, played by a human.
        //  ArcadeMachine.playOneGame(game, level1, recordActionsFile, seed);

        // 2. This plays a game in a level by the controller.
        ArcadeMachine.runOneGame(game, level1, visuals, celdasAgent, recordActionsFile, seed, 0);

        // 3. This replays a game from an action file previously recorded
        //String readActionsFile = recordActionsFile;
        //ArcadeMachine.replayGame(game, level1, visuals, readActionsFile);

        // 4. This plays a single game, in N levels, M times :
//        String level2 = gamesPath + games[gameIdx] + "_lvl" + 1 +".txt";
//        int M = 10;
//        for(int i=0; i<games.length; i++){
//        	game = gamesPath + games[i] + ".txt";
//        	level1 = gamesPath + games[i] + "_lvl" + levelIdx +".txt";
//        	ArcadeMachine.runGames(game, new String[]{level1}, M, sampleMCTSController, null);
//        }

        //5. This starts a game, in a generated level created by a specific level generator

        //if(ArcadeMachine.generateOneLevel(game, randomLevelGenerator, recordLevelFile)){
        //	ArcadeMachine.playOneGeneratedLevel(game, recordActionsFile, recordLevelFile, seed);
        //}

        //6. This plays N games, in the first L levels, M times each. Actions to file optional (set saveActions to true).
//        int N = 82, L = 5, M = 1;
//        boolean saveActions = false;
//        String[] levels = new String[L];
//        String[] actionFiles = new String[L*M];
//        for(int i = 0; i < N; ++i)
//        {
//            int actionIdx = 0;
//            game = gamesPath + games[i] + ".txt";
//            for(int j = 0; j < L; ++j){
//                levels[j] = gamesPath + games[i] + "_lvl" + j +".txt";
//                if(saveActions) for(int k = 0; k < M; ++k)
//                    actionFiles[actionIdx++] = "actions_game_" + i + "_level_" + j + "_" + k + ".txt";
//            }
//            ArcadeMachine.runGames(game, levels, M, sampleMCTSController, saveActions? actionFiles:null);
//        }
    }
}
