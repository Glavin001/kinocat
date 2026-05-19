// Narrow structural shim of the Rapier surface the wrapper uses. Keeps the
// adapter compiling/typechecking without pinning to Rapier's heavy generated
// types; @dimforge/rapier3d(-compat) bodies/worlds satisfy these structurally.

export interface RapierVec3Like {
  x: number;
  y: number;
  z: number;
}

export interface RapierQuatLike {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface RapierBodyLike {
  setTranslation(t: RapierVec3Like, wakeUp: boolean): void;
  setRotation(q: RapierQuatLike, wakeUp: boolean): void;
  setLinvel(v: RapierVec3Like, wakeUp: boolean): void;
  setAngvel(v: RapierVec3Like, wakeUp: boolean): void;
  translation(): RapierVec3Like;
  rotation(): RapierQuatLike;
}

export interface RapierWorldLike {
  timestep: number;
  step(): void;
}
