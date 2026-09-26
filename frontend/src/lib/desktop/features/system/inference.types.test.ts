import { describe, it, expect } from 'vitest';
import { cudaRowState, type CUDABackendStatus } from './inference.types';

function status(overrides: Partial<CUDABackendStatus>): CUDABackendStatus {
  return {
    provider: 'cpu',
    requested: false,
    deviceId: 0,
    libraryPresent: false,
    active: false,
    sessions: [],
    ...overrides,
  };
}

describe('cudaRowState', () => {
  it('is active only when a session runs on CUDA', () => {
    const session = { model: 'm.onnx', deviceId: 0, cudaNodes: 3, totalNodes: 4 };
    expect(
      cudaRowState(
        status({
          provider: 'cuda',
          requested: true,
          libraryPresent: true,
          active: true,
          sessions: [session],
        })
      )
    ).toBe('active');
  });

  it('reports a failed CUDA selection as failed, not available', () => {
    expect(
      cudaRowState(
        status({ provider: 'cuda', requested: true, libraryPresent: true, lastError: 'boom' })
      )
    ).toBe('failed');
  });

  it('reports a CUDA selection with no session and no error as not in use', () => {
    expect(cudaRowState(status({ provider: 'cuda', requested: true, libraryPresent: true }))).toBe(
      'notInUse'
    );
  });

  it('shows availability when CPU is selected', () => {
    expect(cudaRowState(status({ libraryPresent: true }))).toBe('available');
    expect(cudaRowState(status({}))).toBe('unavailable');
  });
});
