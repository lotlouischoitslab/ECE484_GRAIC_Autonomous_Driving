import carla
import math

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        # Parameters for control
        self.desired_speed = 25.0  # m/s (~36 km/h)
        self.threshold_obstacle = 30.0  # Reduced to 20m for earlier reaction
        self.k_obstacle = 25.0  # Increased for stronger avoidance
        self.threshold_boundary = 1.0  # meters
        self.k_boundary = 5.0  # repulsive force constant for boundaries
        self.large_distance = 100.0  # meters for new target point
        self.angle_threshold = math.radians(2)  # Widened to 20 degrees for earlier detection
        self.stopping_distance = 5.0  # meters for hard braking
        self.critical_distance = 3.0  # meters for immediate braking
        self.slowing_distance = 10.0  # Increased to 15m for earlier slowing
        self.sharp_turn_threshold = 0.7  # Steering angle threshold for braking
        # Avoidance lock parameters
        self.avoidance_lock = None  # None, "left", or "right"
        self.lock_duration = 0.5  # 0.5s commitment
        self.lock_timer = 0.0
        self.last_obstacle_dist = float('inf')

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        print("Running Enhanced Agent: Obstacles Ahead Only with Steering Avoidance")

        control = carla.VehicleControl()

        ego_x = transform.location.x
        ego_y = transform.location.y
        ego_yaw = transform.rotation.yaw
        current_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        target_wp = waypoints[0]
        target_x, target_y = target_wp[0], target_wp[1]

        steering_angle = self.calculate_steering_with_avoidance(
            ego_x, ego_y, ego_yaw, target_x, target_y, filtered_obstacles, boundary
        )
        control.steer = steering_angle

        throttle, brake = self.calculate_speed_control(
            ego_x, ego_y, ego_yaw, filtered_obstacles, current_speed, steering_angle, boundary[0], boundary[1], target_wp
        )
        control.throttle = throttle
        control.brake = brake

        return control

    def calculate_steering_with_avoidance(self, ego_x, ego_y, ego_yaw, target_x, target_y, obstacles, boundary):
        self.lock_timer -= 0.033
        if self.lock_timer <= 0:
            self.avoidance_lock = None

        ego_yaw_rad = math.radians(ego_yaw)
        forward_x = math.cos(ego_yaw_rad)
        forward_y = math.sin(ego_yaw_rad)

        force_x = target_x - ego_x
        force_y = target_y - ego_y

        closest_obstacle = None
        min_dist = float('inf')
        for obstacle in obstacles:
            obs_x = obstacle.get_location().x
            obs_y = obstacle.get_location().y
            vec_x = obs_x - ego_x
            vec_y = obs_y - ego_y
            dist = math.hypot(vec_x, vec_y)
            dot = vec_x * forward_x + vec_y * forward_y
            if dist > 0 and dot > 0:
                cos_theta = dot / dist
                if cos_theta > math.cos(self.angle_threshold) and dist < self.threshold_obstacle:
                    if dist < min_dist:
                        min_dist = dist
                        closest_obstacle = (obs_x, obs_y)

        if closest_obstacle:
            obs_x, obs_y = closest_obstacle
            obs_vec_x = obs_x - ego_x
            obs_vec_y = obs_y - ego_y
            dist = math.hypot(obs_vec_x, obs_vec_y)
            
            if dist > self.last_obstacle_dist + 2.0:
                self.avoidance_lock = None
            self.last_obstacle_dist = dist

            cross = forward_x * obs_vec_y - forward_y * obs_vec_x
            if self.avoidance_lock is None:
                avoid_direction = -1 if cross > 0 else 1
                self.avoidance_lock = "left" if avoid_direction == -1 else "right"
                self.lock_timer = self.lock_duration
            else:
                avoid_direction = -1 if self.avoidance_lock == "left" else 1

            # Stronger avoidance, scaled inversely with distance
            avoid_magnitude = self.k_obstacle / max(dist, 0.1)
            avoid_magnitude = min(avoid_magnitude, 15.0)  # Higher cap for sharper turns
            force_x += avoid_magnitude * -forward_y * avoid_direction
            force_y += avoid_magnitude * forward_x * avoid_direction

        left_boundary, right_boundary = boundary[0], boundary[1]
        force_x, force_y = self.add_boundary_forces(
            ego_x, ego_y, left_boundary, force_x, force_y, is_left=True
        )
        force_x, force_y = self.add_boundary_forces(
            ego_x, ego_y, right_boundary, force_x, force_y, is_left=False
        )

        force_magnitude = math.hypot(force_x, force_y)
        if force_magnitude > 0:
            force_x /= force_magnitude
            force_y /= force_magnitude
            if closest_obstacle and dist < self.threshold_obstacle / 2:
                lateral_shift = 10.0 * avoid_direction  # Increased to 10m for wider avoidance
                new_target_x = ego_x + force_x * self.large_distance + lateral_shift * -forward_y
                new_target_y = ego_y + force_y * self.large_distance + lateral_shift * forward_x
            else:
                new_target_x = ego_x + force_x * self.large_distance
                new_target_y = ego_y + force_y * self.large_distance
        else:
            new_target_x, new_target_y = target_x, target_y

        return self.calculate_steering(ego_x, ego_y, ego_yaw, new_target_x, new_target_y)

    def calculate_steering(self, ego_x, ego_y, ego_yaw, target_x, target_y):
        ego_yaw_rad = math.radians(ego_yaw)
        ego_forward_x = math.cos(ego_yaw_rad)
        ego_forward_y = math.sin(ego_yaw_rad)

        wp_vector_x = target_x - ego_x
        wp_vector_y = target_y - ego_y
        wp_distance = math.hypot(wp_vector_x, wp_vector_y)

        if wp_distance > 0:
            wp_vector_x /= wp_distance
            wp_vector_y /= wp_distance

        dot = ego_forward_x * wp_vector_x + ego_forward_y * wp_vector_y
        cross = ego_forward_x * wp_vector_y - ego_forward_y * wp_vector_x
        angle = math.atan2(cross, dot)

        # Allow sharper turns by increasing the denominator
        steering = max(min(angle / (math.pi / 3), 1.0), -1.0)  # 60 degrees max instead of 45
        return steering
        
    def add_boundary_forces(self, ego_x, ego_y, boundary, force_x, force_y, is_left):
        for i in range(len(boundary) - 1):
            s0_x = boundary[i].transform.location.x
            s0_y = boundary[i].transform.location.y
            s1_x = boundary[i + 1].transform.location.x
            s1_y = boundary[i + 1].transform.location.y

            cross = (s1_x - s0_x) * (ego_y - s0_y) - (s1_y - s0_y) * (ego_x - s0_x)
            outside = (cross > 0) if is_left else (cross < 0)

            if outside:
                dx, dy = s1_x - s0_x, s1_y - s0_y
                segment_length_sq = dx * dx + dy * dy
                if segment_length_sq == 0:
                    closest_x, closest_y = s0_x, s0_y
                else:
                    t = max(0, min(1, ((ego_x - s0_x) * dx + (ego_y - s0_y) * dy) / segment_length_sq))
                    closest_x = s0_x + t * dx
                    closest_y = s0_y + t * dy

                dist = math.hypot(ego_x - closest_x, ego_y - closest_y)
                if dist < self.threshold_boundary and dist > 0.01:
                    force_magnitude = self.k_boundary / dist
                    force_x += force_magnitude * (ego_x - closest_x) / dist
                    force_y += force_magnitude * (ego_y - closest_y) / dist

        return force_x, force_y



    def calculate_speed_control(self, ego_x, ego_y, ego_yaw, obstacles, current_speed, steering_angle, left_boundary, right_boundary, target_waypoint):
        # Vehicle's forward vector
        ego_yaw_rad = math.radians(ego_yaw)
        forward_x = math.cos(ego_yaw_rad)
        forward_y = math.sin(ego_yaw_rad)

        # OBSTACLES
        min_dist_front = float('inf')
        for obstacle in obstacles:
            obs_x = obstacle.get_location().x
            obs_y = obstacle.get_location().y
            vec_x = obs_x - ego_x
            vec_y = obs_y - ego_y
            dist = math.hypot(vec_x, vec_y)

            # Only consider obstacles in front
            dot = vec_x * forward_x + vec_y * forward_y
            if dist > 0 and dot > 0:
                cos_theta = dot / dist
                if cos_theta > math.cos(math.radians(30)):
                    min_dist_front = min(min_dist_front, dist)
        
        #WAYPOINTS
        # target_x, target_y = target_waypoint
        # vec_x = target_x - ego_x
        # vec_y = target_y - ego_y
        # dist = math.hypot(vec_x, vec_y)
        # dot = vec_x * forward_x + vec_y * forward_y
        # if dist > 0 and dot > 0:
        #     cos_theta = dot / dist
        #     if cos_theta > math.cos(self.angle_threshold):
        #         min_dist_front = min(min_dist_front, dist)
        #BOUNDARY
        for i in range(len(left_boundary) - 1):  # Iterate through all consecutive boundary points
            s0_x = left_boundary[i].transform.location.x
            s0_y = left_boundary[i].transform.location.y
            s1_x = left_boundary[i + 1].transform.location.x
            s1_y = left_boundary[i + 1].transform.location.y
            
            # Calculate distance and dot product for both points
            vec_x = s0_x - ego_x
            vec_y = s0_y - ego_y
            dist = math.hypot(vec_x, vec_y)
            dot = vec_x * forward_x + vec_y * forward_y
            
            if dist > 0 and dot > 0:
                cos_theta = dot / dist
                if cos_theta > math.cos(self.angle_threshold):
                    # print("here2")
                    min_dist_front = min(min_dist_front, dist)
        
        for i in range(len(right_boundary) - 1):  # Iterate through all consecutive boundary points
            s0_x = right_boundary[i].transform.location.x
            s0_y = right_boundary[i].transform.location.y
            s1_x = right_boundary[i + 1].transform.location.x
            s1_y = right_boundary[i + 1].transform.location.y
            
            # Calculate distance and dot product for both points
            vec_x = s0_x - ego_x
            vec_y = s0_y - ego_y
            dist = math.hypot(vec_x, vec_y)
            dot = vec_x * forward_x + vec_y * forward_y
            
            if dist > 0 and dot > 0:
                cos_theta = dot / dist
                if cos_theta > math.cos(self.angle_threshold):
                    min_dist_front = min(min_dist_front, dist)
        #calculate stopping distance
        # delta_x = -v_i^2/2*a
        print(min_dist_front)
        self.stopping_distance = (-current_speed)**2/(20)
        self.threshold_obstacle = self.stopping_distance
        # print(self.stopping_distance)
        # Speed control logic (only for obstacles ahead)
        if (min_dist_front <= self.stopping_distance + 10) and current_speed < 2:
            return 0.1, 0.0
        elif min_dist_front <= self.stopping_distance + 10:
            return 0.0, 1.0  # Immediate hard brake
        # elif min_dist_front < self.stopping_distance:
        #     return 0.0, 1  # Strong brake
        elif min_dist_front < self.slowing_distance:
            # Gradual slowing
            factor = (min_dist_front - self.stopping_distance) / (self.slowing_distance - self.stopping_distance)
            throttle = 0.5 * factor
            brake = 0.5 * (1 - factor)
        else:
            throttle = 1.0 if current_speed < self.desired_speed else 0.0
            brake = 0.0 if current_speed < self.desired_speed else 0.3

        # Brake for sharp turns
        if abs(steering_angle) > self.sharp_turn_threshold and current_speed > 5.0:
            brake = max(brake, 0.5 * (abs(steering_angle) - self.sharp_turn_threshold) / (1.0 - self.sharp_turn_threshold))
            throttle = min(throttle, 0.5)

        # Ensure throttle and brake are within [0, 1]
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        return throttle, brake
