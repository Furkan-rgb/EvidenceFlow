/** Preserve every selected upload, even when two File objects have identical metadata. */
export function appendFileInstances(selected, files, createInstanceId) {
  return [
    ...selected,
    ...Array.from(files, (file) => ({
      instanceId: createInstanceId(),
      file,
    })),
  ];
}

/** Remove exactly the selected UI instance rather than every metadata duplicate. */
export function removeFileInstance(selected, instanceId) {
  return selected.filter((entry) => entry.instanceId !== instanceId);
}
