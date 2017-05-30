classdef satellite < orbit3d_state
    properties (Access = private)
        mass
        t_len
        t_res
        bool_B
        M_EARTH = 5.972e24;
    end
    
    
    methods
        
        function obj = satellite(mass, tether_len, tether_res, bool_b, ...
                r0, rd0, th0, thd0, phi0, phid0, t0, format)
            obj = obj@orbit3d_state(r0, rd0, th0, thd0, phi0, phid0, t0,...
                format);
            obj.mass = mass;
            obj.t_len = tether_len;
            obj.t_res = tether_res;
            obj.bool_B = bool_b;            
        end
        
        function obj = setmass(m0)
            obj.mass = m0;
        end
        
        function obj = set_tlen(len)
            obj.mass = len;
        end
        
    end
end    