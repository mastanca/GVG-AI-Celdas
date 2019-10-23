package ar.uba.fi.celdas;

import ontology.Types;

public class Theory implements Comparable<Theory> {

    private int k;
    private int p;
    private float utility;
    private char[][] currentState = null;
    private Types.ACTIONS action;
    private char[][] predictedState;

    Theory() {
    }

    Theory(char[][] currentState, Types.ACTIONS action) {
        this.currentState = currentState;
        this.action = action;
        this.predictedState = null;
        this.k = 0;
        this.p = 0;
        this.utility = 0;
    }

    public char[][] getCurrentState() {
        return currentState;
    }

    public void setCurrentState(char[][] currentState) {
        this.currentState = currentState;
    }

    public Types.ACTIONS getAction() {
        return action;
    }

    public void setAction(Types.ACTIONS action) {
        this.action = action;
    }

    public char[][] getPredictedState() {
        return predictedState;
    }

    public void setPredictedState(char[][] predictedState) {
        this.predictedState = predictedState;
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public int getP() {
        return p;
    }

    public void setP(int p) {
        this.p = p;
    }

    public float getUtility() {
        return utility;
    }

    public void setUtility(float utility) {
        this.utility = utility;
    }

    private String charArrayToStr(char[][] charrarray) {
        StringBuilder sb = new StringBuilder();
        if (charrarray != null) {
            for (int i = 0; i < charrarray.length; i++) {
                for (int j = 0; j < charrarray[i].length; j++) {
                    sb.append(charrarray[i][j]);
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    public String actionToString() {
        switch (this.action) {
            case ACTION_NIL:
                return "ACTION_NIL";
            case ACTION_UP:
                return "ACTION_UP";
            case ACTION_LEFT:
                return "ACTION_LEFT";
            case ACTION_DOWN:
                return "ACTION_DOWN";
            case ACTION_RIGHT:
                return "ACTION_RIGHT";
            case ACTION_USE:
                return "ACTION_USE";
            case ACTION_ESCAPE:
                return "ACTION_ESCAPE";
        }
        return "";
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.charArrayToStr(currentState));
        sb.append("\n");
        sb.append(this.actionToString());
        sb.append("\n");
        sb.append(this.charArrayToStr(predictedState));
        return sb.toString();
    }

    @Override
    public int hashCode() {
        return this.toString().hashCode();
    }

    int hashCodeOnlyCurrentState() {
        return this.charArrayToStr(currentState).hashCode();
    }

    int hashCodeOnlyPredictedState() {
        return this.charArrayToStr(predictedState).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Theory other = (Theory) obj;
        return this.toString().equals(other.toString());
    }

    @Override
    public int compareTo(Theory o) {
        if (this.utility == o.utility) {
            float sucessThis = (float) this.p / this.k;
            float sucessOther = (float) o.p / o.k;
            return Math.round((sucessThis - sucessOther) * 100);
        }
        return Math.round((this.utility - o.utility) * 100);
    }

    boolean isWinningTheory() {
        return this.getUtility() > 100;
    }

    public boolean isUseful() {
        return (this.utility > 0);
    }

    public boolean isEqualToCurrentState(char[][] level) {
        return (charArrayToStr(currentState).hashCode() == charArrayToStr(level).hashCode());
    }

    public void addUsedCount() {
        this.k++;
    }

    public void addSuccessCount() {
        this.p++;
    }

    public float getSuccessRate() {
        return (float) this.p / (float) this.k;
    }

    public boolean isComplete() {
        return (currentState != null &&
                action != null &&
                predictedState != null
        );
    }

    public boolean isEqualToPredictedState(char[][] level) {
        return (charArrayToStr(predictedState).hashCode() == charArrayToStr(level).hashCode());
    }
}

