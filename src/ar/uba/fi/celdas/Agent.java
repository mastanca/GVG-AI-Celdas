package ar.uba.fi.celdas;

import com.google.gson.JsonIOException;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Created with IntelliJ IDEA.
 * User: ssamot
 * Date: 14/11/13
 * Time: 21:45
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Agent extends AbstractPlayer {
    /**
     * Random generator for the agent.
     */
    protected Random randomGenerator;
    /**
     * List of available actions for the agent
     */
    protected ArrayList<Types.ACTIONS> actions;


    private Theories theories;
    private Theory currentTheory;
    private Planner planner;
    private TheoryProcessor theoryProcessor;

    /**
     * Public constructor with state observation and time due.
     *
     * @param so           state observation of the current game.
     * @param elapsedTimer Timer for the controller creation.
     */
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer) {
        randomGenerator = new Random();
        actions = so.getAvailableActions();
        theories = new Theories();
        try {
            theories = TheoryPersistant.load();
        } catch (JsonIOException | IOException e) {
            e.printStackTrace();
        }
        this.planner = new Planner(theories);
        this.theoryProcessor = new TheoryProcessor();
    }


    /**
     * Picks an action. This function is called every game step to request an
     * action from the player.
     *
     * @param stateObs     Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        Perception perception = new Perception(stateObs);
        System.out.println(perception.toString());

        this.processCurrentTheory(stateObs);

        currentTheory = new Theory();
        // Get path based on current perception
        List<Theory> savedTheories = loadTheories(perception);
        List<Theory> usefulTheories = savedTheories.stream().filter(Theory::isUseful).collect(Collectors.toList());

        if (!usefulTheories.isEmpty() && !shouldMakeRandomMove() || getPossibleActions(savedTheories).isEmpty()) {
            planNextMove(usefulTheories);
        } else {
            buildRandomTheory(savedTheories, perception);
        }

        return currentTheory.getAction();
    }

    private void processCurrentTheory(StateObservation stateObs) {
        if (currentTheory != null) {
            this.theoryProcessor.processOngoingTheory(stateObs, currentTheory);
            if (!theories.existsTheory(currentTheory)) {
                theories.add(currentTheory);
            }
            if (currentTheory.isUseful()) {
                planner.registerNewTheory(currentTheory);
            }
        }
    }

    private void planNextMove(List<Theory> usefulTheories) {
        for (Theory theory : usefulTheories) {
            if (theory.isWinningTheory()) {
                currentTheory = theory;
                return;
            }
        }
        if (this.theories.victoryIsKnown()) {
            currentTheory = planner.planVictory(usefulTheories); // TODO: Implement me
        } else {
            currentTheory = planner.selectTheory(usefulTheories); // TODO: Implement me
        }
    }

    private void buildRandomTheory(List<Theory> knownTheories, Perception perception) {
        List<Types.ACTIONS> possibleActions = getPossibleActions(knownTheories);
        if (possibleActions.size() > 1) {
            // We have multiple possible actions so we choose 1 randomly
            currentTheory = new Theory(perception.getLevel(), possibleActions.get(randomGenerator.nextInt(possibleActions.size())));
        } else {
            // We only have one possible action
            currentTheory = new Theory(perception.getLevel(), possibleActions.get(0));
        }
    }

    private List<Types.ACTIONS> getPossibleActions(List<Theory> knownTheories) {
        List<Types.ACTIONS> knowActions = knownTheories.stream().map(Theory::getAction).collect(Collectors.toList());
        return actions.stream().filter(action -> !knowActions.contains(action)).collect(Collectors.toList());
    }

    private List<Theory> loadTheories(Perception perception) {
        Theory theory = new Theory();
        theory.setCurrentState(perception.getLevel());
        return theories.getSortedListForCurrentState(theory);
    }

    private boolean shouldMakeRandomMove() {
        // 5 possible actions do nothing, move backwards, move forward, move left, move right
        return randomGenerator.nextInt(actions.size()) > 0;
    }

    public void result(StateObservation stateObs, ElapsedCpuTimer elapsedCpuTimer) {
        boolean gameOver = stateObs.isGameOver();
        this.theoryProcessor.processLastTheory(stateObs, currentTheory);
        if (!theories.existsTheory(currentTheory)) {
            theories.add(currentTheory);
        }
        if (gameOver) {
            try {
                TheoryPersistant.save(theories);
            } catch (JsonIOException | IOException e) {
                e.printStackTrace();
            }
        }
    }
}
