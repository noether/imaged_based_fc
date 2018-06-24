import numpy as np


class agent:
    def __init__(self, label, color, list_nei, p0, l, gain, log_size):
        self.label = label
        self.color = color
        self.list_nei = list_nei # Tuple [label, desired value]
        self.p = p0
        self.l = l
        self.gain = gain

        # Log
        self.log_i = 0
        self.log_P = np.zeros((log_size, 2))
        self.log_E = np.zeros((log_size, len(list_nei)))

    def control_image_based(self, list_agents):
        u = 0
        nei_count = 0
        for o in [(x,y[1]) for x in list_agents for y in self.list_nei if x.label == y[0]]:
            nei = o[0]
            target = o[1]

            zija = self.p - (nei.p - nei.l)
            zijb = self.p - (nei.p + nei.l)
            zija_hat = zija / np.linalg.norm(zija)
            zijb_hat = zijb / np.linalg.norm(zijb)
            errorij = zija_hat.dot(zijb_hat) - target

            u = u - (zija_hat + zijb_hat)*errorij

            self.log_E[self.log_i, nei_count] = errorij
            nei_count = nei_count + 1

        return self.gain*u

    def step_Euler(self, u, dt):
        self.log_P[self.log_i, :] = self.p
        self.p = self.p + u*dt
        self.log_i = self.log_i + 1
