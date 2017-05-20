classdef orb3d_state     %Basic class to store and change orbital state
    properties (Access = private)
        r             %Standard notations of r, theta, phi
        r_dot
        theta
        theta_dot
        phi
        phi_dot
        time
    end
    
    
    methods
        
        function obj = orb3d_state(r0, rd0, th0, thd0, phi0, phid0,...
                t0, format)
            %When no arguments are specified set all param to 0
            %Else assign the required parameters
            if nargin == 0    
                obj.r = 0;
                obj.r_dot = 0;
                obj.theta = 0;
                obj.theta_dot = 0;
                obj.phi = 0;
                obj.phi_dot = 0;
                obj.time = 0;
            elseif (nargin == 7) || (nargin == 8)
                obj.r = r0;
                obj.r_dot = rd0;
                obj.theta = th0;
                obj.theta_dot = thd0;
                obj.phi = phi0;
                obj.phi_dot = phid0;
                obj.time = t0;
                if (nargin == 8 && format == 'xyz')
                    obj.setpos_xyz(r0, th0, phi0)
                    obj.setvel_xyz(rd0, thd0, phid0)
                elseif (format ~= 'sph')
                    disp('Specify either xyz or sph') 
            else
                disp('Specify all arguments or no arguments')
                disp(nargin)
            end
        end
        
        function obj = setstate(obj, state)
            obj.r = state(1);
            obj.r_dot = state(2);
            obj.theta = state(3);
            obj.theta_dot = state(4);
            obj.phi = state(5);
            obj.phi_dot = state(6);
        end
        
        function obj = setpos_sph(obj, pos)
            obj.r = pos(1);
            obj.theta = pos(2);
            obj.phi = pos(3);
        end
        
        function obj = setpos_xyz(obj, pos)
            obj.r = (pos(1)^2 + pos(2)^2 + pos(3)^2)^0.5;
            obj.theta = arccos(pos(3)/obj.r);
            obj.phi = atan2(pos(2), pos(1));
            if obj.phi < 0
                obj.phi = obj.phi + 2*pi;
            end
        end
        
        function obj = setvel_sph(obj, vel)
            obj.r_dot = vel(1);
            obj.theta_dot = vel(2)/obj.r;
            obj.phi_dot = vel(3)/(obj.r*sin(self.theta));
        end
        
        function obj = setvel_xyz(obj, vel)
            posxyz = obj.getpos_xyz();
            obj.r_dot = dot(vel, posxyz)/norm(posxyz);
            obj.theta_dot = (obj.r_dot*posxyz(3) - vel(3)*obj.r)/...
                (obj.r*(obj.r^2 - posxyz(3)^2)^0.5);
            obj.phi_dot = (posxyz(1)*vel(2) - posxyz(2)*vel(1))/...
                (posxyz(1)^2 + posxyz(2)^2);
        end
        
        function param = getstate(obj)
            param(1) = obj.r;
            param(2) = obj.r_dot;
            param(3) = obj.theta;
            param(4) = obj.theta_dot;
            param(5) = obj.phi;
            param(6) = obj.phi_dot;
        end
        
        function time = gettime(obj)
            time = obj.time;
        end
        
        function pos = getpos_sph(obj)
            pos(1) = obj.r;
            pos(2) = obj.theta;
            pos(3) = obj.phi;
        end
        
        function pos = getpos_xyz(obj)
            pos(1) = obj.r*sin(obj.theta)*cos(obj.phi);
            pos(2) = obj.r*sin(obj.theta)*sin(obj.phi);
            pos(3) = obj.r*cos(obj.theta);
        end
        
        function v = getvel_sph(obj)
            v(1) = obj.r_dot;
            v(2) = obj.r*obj.theta_dot;
            v(3) = obj.r*sin(obj.theta)*obj.phi_dot;
        end
        
        function dotdot = getdotdot(obj, a_r, a_t, a_p)
            dotdot(1) = a_r + obj.r*obj.theta_dot^2 ...
                 + obj.r*sin(obj.theta)^2*obj.phi_dot^2;
            dotdot(2) = (a_t - 2*obj.r_dot*obj.theta_dot + ...
                obj.r*obj.phi_dot^2*sin(obj.theta)*cos(obj.theta))/obj.r;
            dotdot(3) = (a_p - 2*obj.r_dot*obj.phi_dot*sin(obj.theta) - ...
                2*obj.r*obj.theta_dot*obj.phi_dot*cos(obj.theta))/...
                    (obj.r*sin(obj.theta));
        end
        
        function obj = rk4_step(obj, dt, acc_f, dis)
            if nargin == 3
                dis = 0;
            end
            param0 = obj.getstate();
            t0 = obj.gettime();
            k = zeros(4, 6);
            for i=1:4
                param = obj.getstate();
                k(i, 1) = param(2)*dt;
                k(i, 3) = param(4)*dt;
                k(i, 5) = param(6)*dt;
                acc = acc_f(obj);
                dotdot = obj.getdotdot(acc(1), acc(2), acc(3));
                if dis == 1
                    disp(dotdot(1))
                end
                k(i, 2) = dotdot(1)*dt;
                k(i, 4) = dotdot(2)*dt;
                k(i, 6) = dotdot(3)*dt;
                if (i <= 2)
                    p_new = param0 + k(i, :)/2;
                    obj = obj.setstate(p_new(1), p_new(2), p_new(3),  ...
                        p_new(4), p_new(5), p_new(6));
                    obj = obj.settime(t0 + dt/2.0);
                elseif (i == 3)
                    p_new = param0 + k(i, :);
                    obj = obj.setstate(p_new(1), p_new(2), p_new(3), ...
                        p_new(4), p_new(5), p_new(6));
                    obj = obj.settime(t0 + dt);
                end
            end
            par_n = param0 + (k(1, :) + 2*k(2, :) + 2*k(3, :) + ...
                k(4, :))/6.0;
            par_n(5) = par_n(5) - floor(par_n(5)/(2*pi))*2*pi;
            obj = obj.setstate(par_n(1), par_n(2), par_n(3), par_n(4), ...
                par_n(5), par_n(6));
            obj = obj.settime(t0 + dt);
        end
        
    end
    
    
    methods (Access = protected)
        function obj = settime(obj, t)
            obj.time = t;
        end
    end
    
end
