//cargo run --example  Linear_ODE_f32 --release

use arrayfire;
use RayBNN_DiffEq;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

fn main() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);

	// Set the Linear Differentail Equation
	// dy/dt = sin(t)
	let diffeq = |t: &arrayfire::Array<f32>, y: &arrayfire::Array<f32>| -> arrayfire::Array<f32> {
		arrayfire::sin(&t) 
	};

	//Start at t=0 and end at t=1000
	//Step size of 0.0001
	//Relative error of 1E-4
	//Absolute error of 1E-4
	//Error Type compute the total error of every element in y
	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f32> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0f32,
		tend: 1000.0f32,
		tstep: 0.0001f32,
		rtol: 1.0E-4f32,
	    atol: 1.0E-4f32,
		error_select: RayBNN_DiffEq::ODE::ODE45::error_type::TOTAL_ERROR
	};

	let t_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut t = arrayfire::constant::<f32>(0.0,t_dims);

	let y0_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut y = arrayfire::constant::<f32>(0.0,y0_dims);
	let mut dydt = arrayfire::constant::<f32>(0.0,y0_dims);

	//Initial Point of Differential Equation
	//Set y(t=0) = 1.0
	let y0 = arrayfire::constant::<f32>(1.0,y0_dims);

	println!("Running");

	arrayfire::sync(DEVICE);
	let starttime = std::time::Instant::now();

	//Run Solver
	RayBNN_DiffEq::ODE::ODE45::linear_ode_solve(
		&y0
		,diffeq
		,&options
		,&mut t
		,&mut y
		,&mut dydt
	);

	arrayfire::sync(DEVICE);

	let elapsedtime = starttime.elapsed();
	
	arrayfire::sync(DEVICE);

	arrayfire::print_gen("y".to_string(), &y,Some(6));
	arrayfire::print_gen("t".to_string(), &t,Some(6));

	println!("Computed {} Steps In: {:.6?}", y.dims()[1],elapsedtime);


	//Error Analysis
	let actualy = 2.0f32 - arrayfire::cos(&t);
	let error = y - actualy;
	//arrayfire::print_gen("error".to_string(), &error,Some(6));
}
