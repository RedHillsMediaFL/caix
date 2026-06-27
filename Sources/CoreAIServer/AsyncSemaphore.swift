import Foundation

/// A minimal FIFO async semaphore used to serialise access to a non-`Sendable` resource
/// (a hot `PersistentModel`) across `await` points without busy-waiting.
///
/// Generation mutates engine state (KV cache, processed-token count) across many suspension
/// points, so two concurrent `generate` calls on the same model must not interleave. Each
/// model handle holds one of these with a single permit and brackets the whole generation in
/// `acquire()` / `release()`. Concurrent callers queue in arrival order and resume one at a
/// time. The guarded work runs in the *caller's* context (only the permit counter lives in
/// this actor), so no non-`Sendable` state crosses an isolation boundary.
actor AsyncSemaphore {
    private var permits: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(permits: Int = 1) { self.permits = permits }

    /// Suspend until a permit is available, then take it.
    func acquire() async {
        if permits > 0 {
            permits -= 1
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    /// Return a permit, resuming the next waiter (if any) instead of incrementing the count.
    func release() {
        if waiters.isEmpty {
            permits += 1
        } else {
            let next = waiters.removeFirst()
            next.resume()
        }
    }
}
