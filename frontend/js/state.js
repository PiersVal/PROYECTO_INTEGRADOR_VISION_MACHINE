/**
 * state.js
 * Responsabilidad: Gestionar el estado global de la aplicación
 * - Almacenar items de galería y resultados de clustering
 * - Estados de métricas externas para cada método
 */

export const state = {
  items: [],
  results: [],
  currentMethod: null, // Trackear método actual
};

export const externalMetricsState = {
  numGroups: 0,
  groupsData: {}
};

export const externalMetricsHuState = {
  numGroups: 0,
  groupsData: {}
};

export const externalMetricsZernikeState = {
  numGroups: 0,
  groupsData: {}
};

export const externalMetricsSiftState = {
  numGroups: 0,
  groupsData: {}
};

export const externalMetricsHogState = {
  numGroups: 0,
  groupsData: {}
};

export const externalMetricsCnnState = {
  numGroups: 0,
  groupsData: {}
};

// Métodos para modificar el estado
export function addItem(item) {
  state.items.push(item);
}

export function addResult(result) {
  state.results.push(result);
}

export function clearResults() {
  state.results = [];
}

export function clearAll() {
  state.items = [];
  state.results = [];
}
