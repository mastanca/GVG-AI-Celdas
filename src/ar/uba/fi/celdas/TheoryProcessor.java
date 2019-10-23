package ar.uba.fi.celdas;

import core.game.StateObservation;

public class TheoryProcessor {

    public void processOngoingTheory(StateObservation stateObs, Theory theory) {
        Perception perception = new Perception(stateObs);
        if (theory.isEqualToCurrentState(perception.getLevel())) {
            this.update(perception, theory, 0);
        } else {
            this.update(perception, theory, 1);
        }
    }

    private void update(Perception perception, Theory theory, float utility) {
        theory.addUsedCount();
        if (theory.isComplete()) {
            processPrediction(theory, perception);
        } else {
            theory.setUtility(utility);
            theory.setPredictedState(perception.getLevel());
            theory.addSuccessCount();
        }
    }

    private void processPrediction(Theory theory, Perception perception) {
        if (theory.isEqualToPredictedState(perception.getLevel())) {
            theory.addSuccessCount();
        }
    }

    public void processLastTheory(StateObservation stateObs, Theory theory) {
        Perception perception = new Perception(stateObs);
        if (stateObs.getGameWinner().toString().equals("PLAYER_WINS")) {
            this.update(perception, theory, 10000);
        } else {
            this.update(perception, theory, -10000);
        }
    }
}
