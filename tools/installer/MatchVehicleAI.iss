; Inno Setup script for the thin (download-at-install) Windows installer.
;
; Compile with (ISCC.exe ships on GitHub's windows runners):
;   ISCC /DAppVersion=1.2.3 tools\installer\MatchVehicleAI.iss
;
; What this produces is around 20 MB against the packaged builds' 350 MB /
; 2.7 GiB -- and almost all of that 20 MB is uv.exe (49 MB on disk), not the
; application, which is about 1 MB of Python. The difference is not
; compression: the dependencies simply are not in here.
; Setup fetches them from PyPI and pytorch.org while it runs, picking the
; CPU or CUDA torch from the machine's own driver instead of asking the user
; to have chosen the right archive. See tools/installer/bootstrap.py for the
; trade-offs that buys and costs.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

#define AppName "Match-Vehicle-AI"
#define AppExeName "MatchVehicleAI.cmd"
; tools\installer\ -> repository root
#define RepoRoot "..\.."

[Setup]
AppId={{8F4C1E23-9B7A-4D62-A5E8-6C3F0B1D7A94}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Warissakorn
AppSupportURL=https://github.com/Warissakorn/Match-Vehicle-AI
DefaultDirName={autopf}\MatchVehicleAI
; Explicit rather than relying on Inno's own default (which is also "no"):
; the wizard's "Select Destination Location" page is a deliberate feature
; here, not an oversight to silently lose if someone later adds a directive
; that turns it off.
DisableDirPage=no
DefaultGroupName={#AppName}
OutputBaseFilename=MatchVehicleAI-windows-installer
OutputDir={#RepoRoot}
SetupIconFile={#RepoRoot}\assets\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Per-user by default, so setup never raises a UAC prompt and the install
; needs no administrator. That fits what actually gets written: every
; user-writable thing the app owns (models, settings, logs, OCR caches)
; already lives under %LOCALAPPDATA% per user, so a machine-wide install
; would share only the read-only half. Users who do want it under Program
; Files can still say so -- the override below turns the choice into a
; dialog rather than removing it.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
; The download is the long pole here, not the disk write: a CUDA environment
; pulls roughly 2.5 GB, and Setup's own progress bar cannot show any of it
; because the work happens in provision.cmd after installation. Saying so on
; the welcome page is the only honest place -- by the time the console
; appears the user has already committed.
WelcomeLabel2=This will install [name/ver] on your computer.%n%nSetup downloads the app's dependencies while it runs, so it needs a working internet connection and can take several minutes -- around 2.5 GB if an NVIDIA GPU is detected, less otherwise. Progress is shown in a console window.%n%nIf you need an offline install, cancel and use MatchVehicleAI-windows.zip from the Releases page instead.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; uv drives the whole provisioning step, so it has to be present before any
; of it runs -- it cannot itself be downloaded by the thing it bootstraps.
Source: "{#RepoRoot}\uv.exe"; DestDir: "{app}"; Flags: ignoreversion

; The application, which is the small part. Everything below is source that
; the environment built at install time runs directly.
Source: "{#RepoRoot}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs
Source: "{#RepoRoot}\src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs
Source: "{#RepoRoot}\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs
Source: "{#RepoRoot}\tools\install_torch.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\pyinstaller_entry.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\installer\bootstrap.py"; DestDir: "{app}\tools\installer"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\installer\provision.cmd"; DestDir: "{app}\tools\installer"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\installer\MatchVehicleAI.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\config.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\cli.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\extract_video.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\models_cli.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\requirements-lock.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    IconFilename: "{app}\assets\icon.ico"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    IconFilename: "{app}\assets\icon.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
; Both directories are created *after* installation, by provision.cmd, so
; Inno has no record of their contents and a plain uninstall would leave
; gigabytes behind. They are listed explicitly for that reason. Nothing the
; user owns is in either -- models, settings and logs live under
; %LOCALAPPDATA%\MatchVehicleAI and are deliberately kept, so reinstalling
; does not re-download ~110 MB of model weights.
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
{ ISPP -- the preprocessor -- runs over this file before the Pascal compiler
  and reads any line whose first non-blank character is '#' as one of its own
  directives. A '#13#10' that begins a continuation line is therefore not a
  character code but an unknown directive, and aborts the compile. Naming the
  pair once keeps that '#' off the start of every line that needs a break. }
const
  NL = #13#10;
  { The literal GUID from [Setup]'s AppId, without the doubled braces that
    section needs to escape Inno's own runtime-constant syntax -- this is a
    plain Pascal string, so a single pair of braces around the GUID is
    correct. Kept as its own constant rather than typed twice: the
    uninstall registry key below and AppId above have to stay byte-for-byte
    the same GUID or the lookup silently finds nothing. Do not quote a
    brace pair inside a brace comment anywhere in this file -- Pascal
    comments do not nest and are not escapable, so the comment closes at
    the next close-brace and leaves the rest as live code. }
  AppUninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{8F4C1E23-9B7A-4D62-A5E8-6C3F0B1D7A94}_is1';

{ Reads the previous installation's uninstaller path, checking both the
  per-user and machine-wide registry locations: PrivilegesRequiredOverridesAllowed
  lets someone choose an admin install (HKLM) even though the default here is
  per-user (HKCU), and an earlier run of this same setup could have gone
  either way. Empty string means "not installed". }
function GetExistingUninstallString(): String;
var
  Value: String;
begin
  Value := '';
  if not RegQueryStringValue(HKLM, AppUninstallKey, 'UninstallString', Value) then
    RegQueryStringValue(HKCU, AppUninstallKey, 'UninstallString', Value);
  Result := Value;
end;

function GetExistingVersion(): String;
var
  Value: String;
begin
  Value := '';
  if not RegQueryStringValue(HKLM, AppUninstallKey, 'DisplayVersion', Value) then
    RegQueryStringValue(HKCU, AppUninstallKey, 'DisplayVersion', Value);
  Result := Value;
end;

{ Detects an existing install up front and offers a real choice instead of
  silently overwriting it: remove the old copy first (a clean upgrade),
  leave it in place and install over it (repair -- reruns the file copy and
  re-provisions the environment, useful if a previous install ended up
  half-broken), or back out untouched. Runs before the wizard's first page
  so the answer decides how installation proceeds, not after the user has
  already clicked through everything. }
function InitializeSetup(): Boolean;
var
  ExistingUninstaller, ExistingVersion, Prompt: String;
  ResultCode, Choice: Integer;
begin
  Result := True;
  ExistingUninstaller := GetExistingUninstallString();
  if ExistingUninstaller = '' then
    Exit;

  ExistingVersion := GetExistingVersion();
  Prompt := 'Match-Vehicle-AI';
  if ExistingVersion <> '' then
    Prompt := Prompt + ' ' + ExistingVersion;
  Prompt := Prompt + ' is already installed.' + NL + NL +
    'Yes    - remove the existing installation first, then install ' +
    '{#AppVersion} cleanly.' + NL +
    'No     - keep the existing files and install {#AppVersion} over them ' +
    '(repair / reinstall in place).' + NL +
    'Cancel - exit Setup without changing anything.';

  Choice := MsgBox(Prompt, mbConfirmation, MB_YESNOCANCEL);
  case Choice of
    IDYES:
      begin
        ExistingUninstaller := RemoveQuotes(ExistingUninstaller);
        if not Exec(ExistingUninstaller, '/VERYSILENT /NORESTART /SUPPRESSMSGBOXES',
                    '', SW_SHOW, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
        begin
          MsgBox('Could not remove the existing installation, so Setup will ' +
                 'exit. You can uninstall it manually from Windows Settings ' +
                 'and run this installer again.', mbError, MB_OK);
          Result := False;
        end;
      end;
    IDCANCEL:
      Result := False;
    { IDNO falls through: keep the existing files, proceed to install/repair
      over them. }
  end;
end;

function ProvisionEnvironment(): Boolean;
var
  ResultCode: Integer;
begin
  { Not runhidden: this downloads and installs gigabytes and can run for
    many minutes. uv's and pip's own progress output in a visible console is
    the only signal the user gets that anything is happening -- a hidden
    window here is indistinguishable from a hung installer. }
  Result := Exec(ExpandConstant('{cmd}'),
                 ExpandConstant('/C ""{app}\tools\installer\provision.cmd" "{app}""'),
                 ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, ResultCode);
  if Result then
    Result := (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not ProvisionEnvironment() then
      { Deliberately fatal rather than a warning that leaves a half-built
        install behind: without its environment the app cannot start at all,
        and an installed-looking Start Menu entry that fails on every click
        is worse than a setup that says it did not finish. }
      { Every part is joined with an explicit '+'. Pascal does not
        concatenate adjacent string literals the way C or Python does, so a
        wrapped message assembled without one is a compile error, not a
        long string. }
      MsgBox('Setup could not download and install the dependencies.' +
             NL + NL +
             'This step needs a working internet connection. The console ' +
             'window that just closed showed what failed.' +
             NL + NL +
             'Match-Vehicle-AI is not usable until setup completes ' +
             'successfully -- please re-run the installer, or use the ' +
             'self-contained MatchVehicleAI-windows.zip from the Releases ' +
             'page instead.',
             mbCriticalError, MB_OK);
  end;
end;
