use arrayfire;

use num_traits;
use half;

const MAGIC: f64 = 2.0;

pub fn add_one<T: arrayfire::FloatingPoint>(x: T, y: &arrayfire::Array<T>) -> arrayfire::Array<T> {
	let newarr = y.clone();

	let zz = vec![MAGIC ];
	let mut b0 = arrayfire::Array::new(&zz, arrayfire::Dim4::new(&[1, 1, 1, 1]));

	let gg:Vec<T> = vec![x ];
	let mut a0 = arrayfire::Array::new(&gg, arrayfire::Dim4::new(&[1, 1, 1, 1]));

	a0 = a0 + b0.cast::<T>();

    return a0 + newarr;
}





const ODE45_C2_f64: f64 = 1.0/5.0;
const ODE45_A21_f64: f64 = 1.0/5.0;

const ODE45_C3_f64: f64 = 3.0/10.0;
const ODE45_A31_f64: f64 = 3.0/40.0;
const ODE45_A32_f64: f64 = 9.0/40.0;

const ODE45_C4_f64: f64 = 4.0/5.0;
const ODE45_A41_f64: f64 = 44.0/45.0;
const ODE45_A42_f64: f64 = -56.0/15.0;
const ODE45_A43_f64: f64 = 32.0/9.0;

const ODE45_C5_f64: f64 = 8.0/9.0;
const ODE45_A51_f64: f64 = 19372.0/6561.0;
const ODE45_A52_f64: f64 = -25360.0/2187.0;
const ODE45_A53_f64: f64 = 64448.0/6561.0;
const ODE45_A54_f64: f64 = -212.0/729.0;

const ODE45_C6_f64: f64 = 1.0;
const ODE45_A61_f64: f64 = 9017.0/3168.0;
const ODE45_A62_f64: f64 = -355.0/33.0;
const ODE45_A63_f64: f64 = 46732.0/5247.0;
const ODE45_A64_f64: f64 = 49.0/176.0;
const ODE45_A65_f64: f64 = -5103.0/18656.0;

const ODE45_C7_f64: f64 = 1.0;
//const ODE45_A71: f64 = 35.0/384.0;
//const ODE45_A72: f64 = 0.0;
//const ODE45_A73: f64 = 500.0/1113.0;
//const ODE45_A74: f64 = 125.0/192.0;
//const ODE45_A75: f64 = -2187.0/6784.0;
//const ODE45_A76: f64 = 11.0/84.0;






const ODE45_B1_f64: f64 = 35.0/384.0;
//const ODE45_B2: f64 = 0.0;
const ODE45_B3_f64: f64 = 500.0/1113.0;
const ODE45_B4_f64: f64 = 125.0/192.0;
const ODE45_B5_f64: f64 = -2187.0/6784.0;
const ODE45_B6_f64: f64 = 11.0/84.0;
//const ODE45_B7: f64 = 0.0;



const ODE45_B1E_f64: f64 = 5179.0/57600.0;
//const ODE45_B2E: f64 = 0.0;
const ODE45_B3E_f64: f64 = 7571.0/16695.0;
const ODE45_B4E_f64: f64 = 393.0/640.0;
const ODE45_B5E_f64: f64 = -92097.0/339200.0;
const ODE45_B6E_f64: f64 = 187.0/2100.0;
const ODE45_B7E_f64: f64 = 1.0/40.0;






pub struct ODE45_Options<Z> {
    pub tstart: Z,
    pub tend: Z,
	pub tstep: Z,
	pub rtol: Z,
	pub atol: Z,
	pub normctrl: bool,
}





//First order system of linear ODE on single device
//initial: initial values of the diffeq in 1 row vector
//diffeq: function that produces derivative at t and x vector
//options: ODE tolerance settings
//Output: Output Spline vector (t_arr,f_arr,dfdt_arr)
pub fn linear_ode_solve<Z: arrayfire::FloatingPoint>(
	initial: &arrayfire::Array<Z>
	,diffeq: impl Fn(&arrayfire::Array<Z>, &arrayfire::Array<Z>) -> arrayfire::Array<Z>
	,options: &ODE45_Options<Z>
	,out_t_arr: &mut arrayfire::Array<Z>
	,out_f_arr: &mut arrayfire::Array<Z>
	,out_dfdt_arr: &mut arrayfire::Array<Z>)
	{

	let var_num: u64 = initial.dims()[1];


	let t_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	//ODE45 Constants
	let temp_constant = vec![ODE45_C2_f64 ];
	let mut ODE45_C2 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A21_f64 ];
	let mut ODE45_A21 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_C3_f64 ];
	let mut ODE45_C3 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A31_f64 ];
	let mut ODE45_A31 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A32_f64 ];
	let mut ODE45_A32 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_C4_f64 ];
	let mut ODE45_C4 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_A41_f64 ];
	let mut ODE45_A41 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A42_f64 ];
	let mut ODE45_A42 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A43_f64 ];
	let mut ODE45_A43 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_C5_f64 ];
	let mut ODE45_C5 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_A51_f64 ];
	let mut ODE45_A51 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A52_f64 ];
	let mut ODE45_A52 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A53_f64 ];
	let mut ODE45_A53 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A54_f64 ];
	let mut ODE45_A54 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_C6_f64 ];
	let mut ODE45_C6 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A61_f64 ];
	let mut ODE45_A61 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A62_f64 ];
	let mut ODE45_A62 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A63_f64 ];
	let mut ODE45_A63 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_A64_f64 ];
	let mut ODE45_A64 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_A65_f64 ];
	let mut ODE45_A65  = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_C7_f64 ];
	let mut ODE45_C7  = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B1_f64 ];
	let mut ODE45_B1 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_B3_f64 ];
	let mut ODE45_B3 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B4_f64 ];
	let mut ODE45_B4  = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B5_f64 ];
	let mut ODE45_B5 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B6_f64 ];
	let mut ODE45_B6 = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_B1E_f64 ];
	let mut ODE45_B1E = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B3E_f64 ];
	let mut ODE45_B3E = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B4E_f64 ];
	let mut ODE45_B4E = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B5E_f64 ];
	let mut ODE45_B5E = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();
	
	let temp_constant = vec![ODE45_B6E_f64 ];
	let mut ODE45_B6E = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ODE45_B7E_f64 ];
	let mut ODE45_B7E = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();








	let mut t: Z = options.tstart.clone()  ;
	let tend: Z =   options.tend.clone()  ;
	let mut tstep: Z =  options.tstep.clone() ;
	let rtol: Z = options.rtol.clone() ;
	let atol: Z = options.atol.clone() ;
	let normctrl: bool = options.normctrl.clone() ;
	let mut cur_point = initial.clone();


	//Calculate derivative k1
	let mut k1 = diffeq(t, &cur_point);

	//Output array
	*out_t_arr = arrayfire::constant::<Z>(t,t_dims);
	*out_f_arr =  initial.clone();
	*out_dfdt_arr = k1.clone();


	let mut nerr: Z = 1.0;
	let mut rerr: Z = 1.0;
	let mut tol: Z = 1.0;

	let cmp_dims = arrayfire::Dim4::new(&[2,var_num,1,1]);
	let mut cmparr = arrayfire::constant::<Z>(t,t_dims);

	if normctrl == false
	{
		cmparr = arrayfire::constant::<Z>(atol,cmp_dims);
	}


	let mut tol_cpu: Vec<Z> = vec![1.0];
	let mut nerr_cpu: Vec<Z> = vec![1.0];




    let mut t2 = t.clone();
    let mut t3 = t.clone();
    let mut t4 = t.clone();
    let mut t5 = t.clone();
    let mut t6 = t.clone();
    let mut t7 = t.clone();



    let mut point2 = cur_point.clone();
    let mut point3 = cur_point.clone();
    let mut point4 = cur_point.clone();
    let mut point5 = cur_point.clone();
    let mut point6 = cur_point.clone();
    let mut point7 = cur_point.clone();




    let mut k2 = k1.clone();
    let mut k3 = k1.clone();
    let mut k4 = k1.clone();
    let mut k5 = k1.clone();
    let mut k6 = k1.clone();
    let mut k7 = k1.clone();


    let mut y0 = k1.clone();
    let mut y1 = k1.clone();
    let mut subtract = k1.clone();


    let mut t_elem = arrayfire::constant::<Z>(t.clone() ,t_dims);
    let mut abserror = arrayfire::constant::<Z>(t.clone() ,t_dims);
    let mut absvec = arrayfire::constant::<Z>(t.clone() ,t_dims);
    let mut minarr = arrayfire::constant::<Z>(t.clone() ,t_dims);
    let mut result = arrayfire::constant::<Z>(t.clone() ,t_dims);
    let mut tol_gpu = arrayfire::constant::<Z>(t.clone() ,t_dims);
    let mut nerr_gpu = arrayfire::constant::<Z>(t.clone() ,t_dims);







	while   t < tend  {

		//Time arr 2
		t2 = t.clone() + (tstep*ODE45_C2) ;

		//Create point vector 2
		point2 = cur_point.clone() +  (tstep*ODE45_A21*(k1.clone()) );

		//Calculate derivative k2
		k2 = diffeq(t2, &point2);






		//Time arr 3
		t3 = t.clone()  + (tstep*ODE45_C3) ;

		//Create point vector 3
		point3 = cur_point.clone() +  tstep*(  (ODE45_A31*(k1.clone()))   +  (ODE45_A32*(k2.clone()))    );

		//Calculate derivative k3
		k3 = diffeq(t3,&point3) ;







		//Time arr 4
		t4 = t.clone() + (tstep*ODE45_C4);

		//Create point vector 4
		point4 = cur_point.clone() +  tstep*(  (ODE45_A41*(k1.clone()))   +  (ODE45_A42*(k2.clone()))  +  (ODE45_A43*(k3.clone()))   );


		//Calculate derivative k4
		k4 = diffeq(t4,&point4);






		//Time arr 5
		t5 = t.clone() + (tstep*ODE45_C5) ;

		//Create point vector 4
		point5 = cur_point.clone() +  tstep*(  (ODE45_A51*(k1.clone()))   +  (ODE45_A52*(k2.clone()))  +  (ODE45_A53*(k3.clone()))      +  (ODE45_A54*(k4.clone()))      );

		//Calculate derivative k5
		k5 = diffeq(t5,&point5);








		//Time arr 5
		t6 = t.clone() + (tstep*ODE45_C6);

		//Create point vector 4
		point6 = cur_point.clone() +  tstep*(  (ODE45_A61*(k1.clone()))   +  (ODE45_A62*(k2.clone()))  +  (ODE45_A63*(k3.clone()))      +  (ODE45_A64*(k4.clone()))   +  (ODE45_A65*(k5.clone()))     );

		//Calculate derivative k4
		k6 = diffeq(t6,&point6);






		//Time arr 5
		t7 = t.clone() + (tstep*ODE45_C7);

		//Create point vector 4
		//let point7 = cur_point.clone() +  tstep*(  (ODE45_A71*(k1.clone()))   +  (ODE45_A72*(k2.clone()))  +  (ODE45_A73*(k3.clone()))      +  (ODE45_A74*(k4.clone()))   +  (ODE45_A75*(k5.clone()))    +  (ODE45_A76*(k6.clone()))   );

		y0 = tstep*( (ODE45_B1*k1.clone())  +   (ODE45_B3*k3.clone()) +  (ODE45_B4*k4.clone()) +  (ODE45_B5*k5.clone()) +  (ODE45_B6*k6.clone())  );
		point7 = cur_point.clone() + y0.clone();

		//Calculate derivative k4
		k7 = diffeq(t7,&point7);




		y1 = tstep*( (ODE45_B1E*k1.clone())   +  (ODE45_B3E*k3) +  (ODE45_B4E*k4) +  (ODE45_B5E*k5) +  (ODE45_B6E*k6) +  (ODE45_B7E*k7.clone()) );
		subtract = y1.clone() - y0.clone();


		if normctrl
		{
			nerr = arrayfire::norm::<Z>(&subtract,arrayfire::NormType::VECTOR_2,0.0,0.0  ) as Z  ;
			rerr = arrayfire::norm::<Z>(&y0,arrayfire::NormType::VECTOR_2,0.0,0.0  )  as Z;
			tol = atol.min( rtol*rerr );
		}
		else
		{
			abserror = arrayfire::abs(&subtract);
			absvec = rtol * arrayfire::abs(&y0);

			arrayfire::set_row(&mut cmparr, &absvec,1);
			minarr = arrayfire::min(&cmparr,0);
			result = abserror.clone() - minarr.clone();


            let (_,_,idx) = arrayfire::imax_all(&result);

            tol_gpu = arrayfire::col(&minarr, idx as i64);
            nerr_gpu = arrayfire::col(&abserror, idx as i64);





			tol_gpu.host(&mut tol_cpu);
			tol = tol_cpu[0];


			nerr_gpu.host(&mut nerr_cpu);
			nerr = nerr_cpu[0];
		}




		if  nerr < tol
		{
			//New point
			cur_point = point7.clone();

			//New time
			t = t7.clone();

			//New derivative
			k1 = k7.clone();

			t_elem = arrayfire::constant::<Z>(t.clone() ,t_dims);

			//Save to array
			*out_t_arr = arrayfire::join::<Z>(0,out_t_arr,&t_elem);

			*out_f_arr = arrayfire::join::<Z>(0,out_f_arr,&cur_point);

			*out_dfdt_arr = arrayfire::join::<Z>(0,out_dfdt_arr,&k1);

		}


		tstep = 0.9*tstep*( ( ( (tol/(nerr + 1E-30)).powf(0.2)).max(0.1)  ).min(10.0)  );

	}


}
