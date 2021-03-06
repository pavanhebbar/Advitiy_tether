classdef satellite < orbit3d_state
    properties (Access = private)
        mass
        inertia
        t_len
        t_res
        main_m
        bool_B
        
    end
    
    
    methods
        
        function obj = satellite(mass, I_matrix, bool_b, tether, ...
                r0, v_r, th0, v_t0, phi0, v_p0, t0, format, main_mass)
            obj = obj@orbit3d_state(r0, v_r, th0, v_t0, phi0, v_p0, t0,...
                format);            
            global G 
            G = 6.67408e-11;
            global M_EARTH 
            M_EARTH = 5.972e24;
            obj.mass = mass;
            obj.inertia = I_matrix;
            obj.teth = tether;
            obj.bool_B = bool_b;
            obj.main_m = M_EARTH;
            if nargin == 13
                obj.main_m = main_mass;
            end
        end
        
        function obj = setmainmass(obj, m0)
            obj.main_m = m0;
        end
        
        function obj = setmass(obj, m0)
            obj.mass = m0;
        end
        
        function obj = set_tether(tether)
            obj.teth = tether;
        end
        
        function obj = setI(I_matrix)
            obj.inertia = I_matrix;
        end
        
        function main_mass = getmainmass(obj)
            main_mass = obj.main_m;
        end
        
        function mass1 = getmass(obj)
            mass1 = obj.mass;
        end
        
        function tether = gettether(obj)
            tether = obj.teth;
        end
        
        function I_matrix = getI(obj)
            I_matrix = obj.inertia;
        end
        
        function coord = getlatlong(obj)
            pos = obj.getpos_sph();
            t = obj.gettime();
            lat = pi/2.0 - pos(2);
            lon = pos(3) - 2*pi/86164.09164*t;
            coord = [lat, lon];
        end
        
        function energy = geten(obj)
            vel = obj.getvel_sph();
            pos = obj.getpos_sph();
            energy = 0.5*obj.mass*norm(vel)^2-G*obj.main_m*obj.mass/pos(1);
        end
        
        function a = get_a(obj)
            energy = obj.geten();
            a = -1*G*obj.main_m*obj.mass/(2*energy);
        end
        
        function e_vec = getecc(obj)
            mu = G*(obj.main_m + obj.mass);
            h = obj.sp_ang_mom();
            v = obj.getvel_xyz();
            r = obj.getpos_xyz();
            e_vec = 1.0/mu*(cross(v, h) - mu*r/norm(r));
        end
        
        function small_omega = arg_per(obj)
            h_vec = obj.sp_ang_mom();
            nvec = cross([0, 0, 1], h_vec);
            evec = obj.getecc();
            small_omega = arccos(dot(nvec, evec)/(norm(nvec)*norm(evec)));
            if evec(3) < 0
                small_omega = 2*pi - small_omega;
            end
        end
        
        function anom = true_an(obj)
            v = obj.getpos_sph();
            evec = obj.getecc();
            r = obj.getpos_xyz();
            anom = arccos(dot(evec, r)/(norm(evec)*norm(r)));
            if v(1) < 0
                anom = 2*pi - anom;
            end
        end
        
        function elems = orb_elem(obj)
            h_vec = obj.sp_ang_mom();
            elems(1) = norm(h_vec);
            elems(2) = obj.get_inc();
            elems(3) = obj.get_ascnode();
            evec = obj.getecc();
            elems(4) = norm(evec);
            elems(5) = obj.arg_per();
            elems(6) = obj.true_an();
        end           
            
        
    end
end    