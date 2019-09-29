package ar.uba.fi.celdas;

import java.util.ArrayList;
import java.util.List;

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
public class Planner {
    private Theories theories;
    private List<Theory> currentPlan;

    public Planner(Theories theories) {
        this.theories = theories;
        currentPlan = new ArrayList<>();
    }
}
