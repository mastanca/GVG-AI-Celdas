package ar.uba.fi.celdas;

import java.util.*;

/**
 * ENTRADA: Situación actual, Conjunto de Teorías
 * SALIDA: Plan
 * Armar Pila de Situaciones Deseables (De mayor a menor utilidad)
 * Mientras (Existan Situaciones deseables y no exista situación objetivo)
 * Tomar situación deseable (Raíz)
 * Armar arbol de situaciones
 * Buscar Nodo asociado a la situación actual
 * Si No existe Nodo buscar camino de mínimo recorrido
 * Si no Existe situación objetivo:
 * Tomar plan por contingencia
 * SINO
 * Armar plan asociado a camino mínimo de recorrido
 */
class Planner {
    private Theories theories;
    private List<Theory> currentPlan;
    private Map<Integer, Integer> statesAlredyProcessed;
    private Integer lastState;
    private Random randomGenerator;

    public Planner(Theories theories) {
        this.randomGenerator = new Random();
        this.theories = theories;
        this.currentPlan = new ArrayList<>();
        this.statesAlredyProcessed = new HashMap<>();
    }

    public Theory planVictory(List<Theory> usefulTheories) {
        Theory theoryPreceeded = this.getTheoryPreceeded(usefulTheories.get(0).hashCodeOnlyCurrentState());

        if (theoryPreceeded == null) {
            this.buildPlan(usefulTheories);
            theoryPreceeded = this.getTheoryPreceeded(usefulTheories.get(0).hashCodeOnlyCurrentState());
        }

        if (theoryPreceeded == null) {
            return this.selectTheory(usefulTheories);
        }

        return theoryPreceeded;
    }

    private void buildPlan(List<Theory> usefulTheories) {
        List<Theory> winningTheories = theories.getWinningTheories();
        int currentState = usefulTheories.get(0).hashCodeOnlyCurrentState();
        for (Theory theory : winningTheories) {
            List<Integer> statesAlreadyCovered = new ArrayList<>();
            statesAlreadyCovered.add(theory.hashCodeOnlyCurrentState());
            currentPlan = buildPath(currentState, statesAlreadyCovered, theory.hashCodeOnlyCurrentState());
        }
    }

    private List<Theory> buildPath(int currentState, List<Integer> statesAlreadyCovered, int state) {
        List<Theory> theoriesChain = this.filterUsefulTheories(theories.getSortedListByPredictedState(state), statesAlreadyCovered);
        List<Theory> path = new ArrayList<>();
        for (Theory theory : theoriesChain) {
            if (theory.hashCodeOnlyCurrentState() == currentState) {
                path.add(theory);
                return path;
            }
            statesAlreadyCovered.add(theory.hashCodeOnlyCurrentState());
            List<Theory> plan = buildPath(currentState, statesAlreadyCovered, theory.hashCodeOnlyCurrentState());
            if (plan != null) {
                path = plan;
                path.add(theory);
                return path;
            }
        }
        return null;
    }

    private List<Theory> filterUsefulTheories(List<Theory> sortedListByPredictedState, List<Integer> statesAlreadyCovered) {
        List<Theory> finalTheories = new ArrayList<>();
        for (Theory theory : sortedListByPredictedState) {
            if (theory.isUseful() && !statesAlreadyCovered.contains(theory.hashCodeOnlyCurrentState())) {
                finalTheories.add(theory);
            }
        }
        return finalTheories;
    }

    private Theory getTheoryPreceeded(int hashCodeOnlyCurrentState) {
        for (Theory theory : currentPlan) {
            if (theory.hashCodeOnlyCurrentState() == hashCodeOnlyCurrentState) {
                return theory;
            }
        }
        return null;
    }

    public Theory selectTheory(List<Theory> usefulTheories) {
        if (usefulTheories.size() < 2) {
            return usefulTheories.get(0);
        }
        float sum = 0;
        for (Theory theory : usefulTheories) {
            sum += this.getScoreFrom(theory);
        }
        int index = randomGenerator.nextInt(Math.round(sum - 1));
        int counter = 0;
        for (Theory theory : usefulTheories) {
            int theoryChances = this.getScoreFrom(theory);
            if ((index >= counter) && (index < counter + theoryChances)) {
                return theory;
            }
            counter += theoryChances;
        }
        return usefulTheories.get(0);
    }

    private int getScoreFrom(Theory theory) {
        int key = theory.hashCodeOnlyPredictedState();
        int modifier = 1;
        if (this.statesAlredyProcessed.containsKey(key)) {
            modifier = this.statesAlredyProcessed.get(key);
            if (modifier > 8) {
                modifier = 8;
            }
        }
        if ((lastState != null) && (key == lastState)) {
            modifier = modifier * 3;
        }
        return Math.round((theory.getUtility() * 1000 * theory.getSuccessRate()) / modifier);
    }

    public void registerNewTheory(Theory currentTheory) {
        int key = currentTheory.hashCodeOnlyCurrentState();
        if (!statesAlredyProcessed.containsKey(key)) {
            statesAlredyProcessed.put(key, 1);
        }
        lastState = key;
        this.updateStates();
    }

    private void updateStates() {
        statesAlredyProcessed.keySet().forEach(key -> {
            statesAlredyProcessed.put(key, statesAlredyProcessed.get(key) + 1);
        });
    }
}
