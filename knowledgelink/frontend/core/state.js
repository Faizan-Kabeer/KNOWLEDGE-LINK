export const state = {
  selectedEntity:   null,   // { id, name }
  selectedRelation: null,   // { id, name }
  allRelations:     [],
  graphData:        { nodes: [], links: [] },
  predictions:      [],
  explainData:      null,
  activePredict:    null,   // index in predictions
};
