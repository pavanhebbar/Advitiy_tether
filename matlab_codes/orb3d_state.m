classdef orb3d_state     % Basic class to store and change orbital state
    properties (Access = private)
        r             % Standard notations of r, theta, phi etc. in ECI
        v_r           %private variables cannot be accessed from outside
        theta
        v_theta
        phi
        v_phi
        time
    end
    
    
    methods
        
        function obj = orb3d_state(r0, vr0, th0, vt0, phi0, vphi0,...
                t0, format)
            %When no arguments are specified set all param to 0
            %Else assign the required parameters
            if nargin == 0    
                obj.r = 0;
                obj.v_r = 0;
                obj.theta = 0;
                obj.v_theta = 0;
                obj.phi = 0;
                obj.v_phi = 0;
                obj.time = 0;
            elseif (nargin == 7) || (nargin == 8)
                obj.r = r0;
                obj.v_r = vr0;
                obj.theta = th0;
                obj.v_theta = vt0;
                obj.phi = phi0;
                obj.v_phi = vphi0;
                obj.time = t0;
                if (nargin == 8 && strcmp(format,'xyz'))
                    obj.setpos_xyz(r0, th0, phi0)
                    obj.setvel_xyz(rd0, thd0, phid0)
                elseif (xor(strcmp(format,'sph'), 1))
                    disp('Specify either xyz or sph') 
                end
            else
                disp('Specify all arguments or no arguments')
                disp(nargin)
            end
        end
        
        function obj = setstate(obj, state)
            obj.r = state(1);
            obj.v_r = state(2);
            obj.theta = state(3);
            obj.v_theta = state(4);
            obj.phi = state(5);
            obj.v_phi = state(6);
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
            obj.v_r = vel(1);
            obj.v_theta = vel(2);
            obj.v_phi = vel(3);
        end
        
        function obj = setvel_xyz(obj, vel)
            posxyz = obj.getpos_xyz();
            obj.v_r = dot(vel, posxyz)/norm(posxyz);
            obj.v_theta = (obj.v_r*posxyz(3) - vel(3)*obj.r)/...
                (obj.r^2 - posxyz(3)^2)^0.5;
            obj.v_phi = (posxyz(1)*vel(2) - posxyz(2)*vel(1))/...
                ((posxyz(1)^2 + posxyz(2)^2)^0.5);
        end
        
        function param = getstate(obj)
            param(1) = obj.r;
            param(2) = obj.v_r;
            param(3) = obj.theta;
            param(4) = obj.v_theta;
            param(5) = obj.phi;
            param(6) = obj.v_phi;
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
            v(1) = obj.v_r;
            v(2) = obj.v_theta;
            v(3) = obj.v_phi;
        end
        
        function v = getvel_xyz(obj)
            v(1) = (obj.v_r*sin(obj.theta)*cos(obj.phi) + ...
                obj.v_theta*cos(obj.theta)*cos(obj.phi) - ...
                obj.v_phi*sin(obj.phi));
            v(2) = (obj.v_r*sin(obj.theta)*sin(obj.phi) + ...
                obj.v_theta*cos(obj.theta)*sin(obj.phi) + ...
                obj.v_phi*cos(obj.phi));
            v(3) = obj.v_r*cos(obj.theta) - obj.v_theta*sin(obj.theta);
        end
            
        
        function dotdot = getdotdot(obj, a_r, a_t, a_p)
            dotdot(1) = a_r + (obj.v_theta^2 + obj.v_phi^2)/obj.r;
            dotdot(2) = a_t - (obj.v_r*obj.v_theta -...
                obj.v_phi^2/tan(obj.theta))/obj.r ;
            dotdot(3) = a_p - (obj.v_r*obj.v_phi + ...
                obj.v_theta*obj.v_phi/tan(obj.theta))/obj.r;
        end
        
        function obj = rk4_step(obj, dt, acc_f)
            param0 = obj.getstate();
            t0 = obj.gettime();
            k = zeros(4, 6);
            for i=1:4
                param = obj.getstate();
                k(i, 1) = param(2)*dt;
                k(i, 3) = param(4)/param(1)*dt;
                k(i, 5) = param(6)/(param(1)*sin(param(2)))*dt;
                acc = acc_f(obj);
                dotdot = obj.getdotdot(acc(1), acc(2), acc(3));
                k(i, 2) = dotdot(1)*dt;
                k(i, 4) = dotdot(2)*dt;
                k(i, 6) = dotdot(3)*dt;
                if (i <= 2)
                    p_new = param0 + k(i, :)/2;
                    obj = obj.setstate(p_new);
                    obj = obj.settime(t0 + dt/2.0);
                elseif (i == 3)
                    p_new = param0 + k(i, :);
                    obj = obj.setstate(p_new);
                    obj = obj.settime(t0 + dt);
                end
            end
            par_n = param0 + (k(1, :) + 2*k(2, :) + 2*k(3, :) + ...
                k(4, :))/6.0;
            par_n(5) = par_n(5) - floor(par_n(5)/(2*pi))*2*pi;
            obj = obj.setstate(par_n);
            obj = obj.settime(t0 + dt);
        end
        
        function h_vec = sp_ang_mom(obj)
            %Return the specific angular momentum of system
            pos = obj.getpos_xyz();
            vel = obj.getvel_xyz();
            h_vec = cross(pos, vel);
        end
        
        function inc = get_inc(obj)
            %Return the inclination of orbit
            h_vec = obj.sp_ang_mom();
            inc = arccos(h_vec(3)/norm(h_vec));
        end
        
        function cap_omega = get_ascnode(obj)
            %Return the longitude of ascending node
            h_vec = obj.sp_ang_mom();
            cap_omega = atan2(h_vec(1), -1*h(2));
            if cap_omega < 0
                cap_omega = cap_omega + 2*pi;
            end
        end
        
    end
    
    
    methods (Access = protected)
        function obj = settime(obj, t)
            obj.time = t;
        end
    end
    
end
