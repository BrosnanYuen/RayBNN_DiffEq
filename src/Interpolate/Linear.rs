use arrayfire;





//t: times values of function f
//f: values of function f at time t
//dfdt:  derivatives of function f at time t
//s: time vector to sample
//Output: values of function f at time s
pub fn run<Z: arrayfire::FloatingPoint>(
	t: &arrayfire::Array<Z>
	,f: &arrayfire::Array<Z>
	,dfdt: &arrayfire::Array<Z>
	,s: &arrayfire::Array<Z>)
	-> arrayfire::Array<Z>  
	{


	let t_dims = t.dims();
	let input_size = t_dims[1];

	let s_dims = s.dims();
	let target_size = s_dims[1];

	let f_dims = f.dims();
	let function_size = f_dims[0];


	let s_arr = arrayfire::transpose(&s,false);

	let mut dist = s_arr - t.clone();

	dist = arrayfire::abs(&dist).cast::<Z>();
	let (_,idx) = arrayfire::imin(&dist,1);
	drop(dist);



	let t_init =  arrayfire::lookup(t, &idx, 1);
	let f_init =  arrayfire::lookup(f, &idx, 1);
	let dfdt_init =  arrayfire::lookup(dfdt, &idx, 1);

	let mut step = s-t_init;



	let step_dims = arrayfire::Dim4::new(&[function_size,1,1,1]);
	step = arrayfire::tile(&step,step_dims);




	let result = f_init + arrayfire::mul(&dfdt_init, &step, false);

	result
}


