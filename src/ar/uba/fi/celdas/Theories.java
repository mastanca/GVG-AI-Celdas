package ar.uba.fi.celdas;

import java.util.*;

public class Theories {

    private Map<Integer, List<Theory>> theoriesForCurrentState;
    private Map<Integer, List<Theory>> theoriesForPredictedState;
    private List<Theory> winningTheories;
    private Set<Integer> existenceSet;

    public Theories() {
        this.theoriesForCurrentState = new HashMap<>();
        this.theoriesForPredictedState = new HashMap<>();
        this.winningTheories = new ArrayList<>();
        this.existenceSet = new HashSet<>();
    }

    public void add(Theory theory) {
        if (!existsTheory(theory)) {
            List<Theory> currentTheories = this.theoriesForCurrentState.computeIfAbsent(theory.hashCodeOnlyCurrentState(), k -> new ArrayList<>());
            List<Theory> predictedTheories = this.theoriesForPredictedState.computeIfAbsent(theory.hashCodeOnlyPredictedState(), k -> new ArrayList<>());
            currentTheories.add(theory);
            predictedTheories.add(theory);

            if (theory.isWinningTheory()) {
                winningTheories.add(theory);
            }

            this.existenceSet.add(theory.hashCode());
        }
    }

    public boolean existsTheory(Theory theory) {
        return this.existenceSet.contains(theory.hashCode());
    }

    public List<Theory> getSortedListForCurrentState(Theory theory) {

        List<Theory> theoryList = this.theoriesForCurrentState.get(theory.hashCodeOnlyCurrentState());
        if (theoryList == null) {
            theoryList = new ArrayList<>();
        }
        Collections.sort(theoryList);
        return theoryList;
    }

    public Set<Integer> getExistenceSet() {
        return existenceSet;
    }

    public void setExistenceSet(Set<Integer> existenceSet) {
        this.existenceSet = existenceSet;
    }

    public Map<Integer, List<Theory>> getTheoriesForCurrentState() {
        return theoriesForCurrentState;
    }

    public void setTheoriesForCurrentState(Map<Integer, List<Theory>> theoriesForCurrentState) {
        this.theoriesForCurrentState = theoriesForCurrentState;
    }

    public Map<Integer, List<Theory>> getTheoriesForPredictedState() {
        return theoriesForPredictedState;
    }

    public void setTheoriesForPredictedState(Map<Integer, List<Theory>> theoriesForPredictedState) {
        this.theoriesForPredictedState = theoriesForPredictedState;
    }

    public List<Theory> getWinningTheories() {
        return winningTheories;
    }

    public void setWinningTheories(List<Theory> winningTheories) {
        this.winningTheories = winningTheories;
    }

    public boolean victoryIsKnown() {
        return this.winningTheories.size() > 0;
    }

    public List<Theory> getSortedListByPredictedState(int state) {
        List<Theory> theoryList = this.theoriesForPredictedState.get(state);
        if (theoryList == null) {
            theoryList = new ArrayList<>();
        }
        Collections.sort(theoryList);
        return theoryList;
    }
}
