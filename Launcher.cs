using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class Program
{
    [STAThread]
    static void Main(string[] args)
    {
        string appDir = AppDomain.CurrentDomain.BaseDirectory;
        string pythonw = Path.Combine(appDir, @".venv\Scripts\pythonw.exe");
        string python = Path.Combine(appDir, @".venv\Scripts\python.exe");
        string script = Path.Combine(appDir, "main_gui.py");

        string exeToUse = File.Exists(pythonw) ? pythonw : python;

        if (!File.Exists(exeToUse))
        {
            MessageBox.Show(
                "Could not find Python environment in .venv!\nMake sure ClipMore is running inside its folder.",
                "ClipMore Studio",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return;
        }

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = exeToUse,
            Arguments = "\"" + script + "\"",
            WorkingDirectory = appDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden
        };

        try
        {
            Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            MessageBox.Show("Failed to launch ClipMore:\n" + ex.Message, "ClipMore Studio", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
