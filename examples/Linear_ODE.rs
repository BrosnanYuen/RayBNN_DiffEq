
use arrayfire;
use RayBNN_DiffEq;

use num_traits;
use half;




const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


fn main() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);

	let diffeq = |t: &arrayfire::Array<f64>, y: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		arrayfire::cos(&t) 
	};

	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f64> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0f64,
		tend: 1000.0f64,
		tstep: 0.001f64,
		rtol: 1.0E-9f64,
	    atol: 1.0E-9f64,
		normctrl: false
	};

	println!("Running");

	arrayfire::sync(0);
	let starttime = std::time::Instant::now();

	let y0_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let mut t = arrayfire::constant::<f64>(0.0,y0_dims);
	let mut y = arrayfire::constant::<f64>(0.0,y0_dims);
	let mut dydt = arrayfire::constant::<f64>(0.0,y0_dims);

	let mut y0 = arrayfire::constant::<f64>(0.0,y0_dims);

	RayBNN_DiffEq::ODE::ODE45::linear_ode_solve(
		&y0
		,diffeq
		,&options
		,&mut t
		,&mut y
		,&mut dydt
	);

	arrayfire::sync(0);

	let elapsedtime = starttime.elapsed();
	
	arrayfire::sync(0);

	arrayfire::print_gen("y".to_string(), &y,Some(6));
	arrayfire::print_gen("t".to_string(), &t,Some(6));

	println!("Computed {} Steps In: {:.6?}", y.dims()[0],elapsedtime);

}
