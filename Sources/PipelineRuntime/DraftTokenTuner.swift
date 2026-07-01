import Foundation

struct DraftTokenTuner {
    let minimum: Int
    let maximum: Int
    private(set) var current: Int
    private var strongAcceptanceStreak = 0

    init(initial: Int, maximum: Int, minimum: Int = 1) {
        let floor = max(1, minimum)
        let ceiling = max(floor, maximum)
        self.minimum = floor
        self.maximum = ceiling
        self.current = min(max(initial, floor), ceiling)
    }

    mutating func observe(accepted: Int, drafted: Int) {
        guard drafted > 0 else { return }
        let accepted = max(0, min(accepted, drafted))
        let ratio = Double(accepted) / Double(drafted)

        if accepted == drafted {
            strongAcceptanceStreak += 1
            if strongAcceptanceStreak >= 2, current < maximum {
                current += 1
                strongAcceptanceStreak = 0
            }
            return
        }

        strongAcceptanceStreak = 0
        if ratio < 0.5, current > minimum {
            current -= 1
        }
    }
}
