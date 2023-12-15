import numpy as np

class CubicPathPlanner:
    def __init__(self, q_start:list, q_finish:list, t_start:float, t_finish:float, q_via:list=[], t_via:list=[])->None:
        self.n_joints = len(q_start)
        self.t_start = t_start
        self.t_finish = t_finish
        self.t_via = t_via.copy()
        
        self.n_via_points = len(q_via)

        self.coeff = np.zeros((self.n_joints, 4, self.n_via_points+1)) 

        for i in range(0, self.n_via_points+1):
            if self.n_via_points == 0 :
                q_s = q_start
                q_f = q_finish
                t_s = t_start
                t_f = t_finish
            elif i == 0:
                q_s = q_start
                q_f = q_via[0]
                t_s = t_start
                t_f = t_via[0]
            elif i == self.n_via_points:
                q_s =  q_via[self.n_via_points-1]
                q_f = q_finish
                t_s = t_via[self.n_via_points-1]
                t_f = t_finish
            else:
                q_s = q_via[i]
                q_f = q_via[i+1]
                t_s = t_via[i]
                t_f = t_via[i+1]
             
            motion_time = t_f - t_s
            for joint_idx in range(self.n_joints):
                self.coeff[joint_idx][0][i] = q_s[joint_idx]
                self.coeff[joint_idx][1][i] = 0
                self.coeff[joint_idx][2][i] = 3*(q_f[joint_idx] - q_s[joint_idx])/(motion_time**2)
                self.coeff[joint_idx][3][i] = - 2*(q_f[joint_idx] - q_s[joint_idx])/(motion_time**3)

    def calc_configuration(self, time:float)->list:
        if self.n_via_points == 0:
            t_s = self.t_start
            coeff_idx = 0
        elif time < self.t_via[0]:
            coeff_idx = 0
            t_s =self.t_start  
        elif time >= self.t_via[-1]:
            coeff_idx = self.n_via_points 
            t_s = self.t_via[-1]
        else:
            for i in range(0, self.n_via_points-1):
                if self.t_via[i] <= time < self.t_via[i+1]:
                    t_s = self.t_via[i]
                    coeff_idx = i

        q_curr = [0.0] * self.n_joints
        for i in range(self.n_joints):
            q_curr[i] = self.coeff[i][0][coeff_idx] + self.coeff[i][1][coeff_idx]*(time-t_s) +self.coeff[i][2][coeff_idx]*(time-t_s)**2 + \
                self.coeff[i][3][coeff_idx]*(time-t_s)**3
        return q_curr
